import inspect
import logging
import os
import sys
import time

from abc import ABC, abstractmethod
from argparse import Namespace
from collections import defaultdict
from queue import Queue
from threading import Thread
from typing import List, Tuple

# output save style
OUTPUT_STYLE_JSON = 'json'
OUTPUT_STYLE_ANNO = 'anno'
OUTPUT_STYLE_ARANGO = 'arango'

# default output file
OUTPUT_DEFAULT = './annotations'


class AggQueue:
    def __init__(self, q_agg):
        self.q_agg = q_agg
    
    def put(self, item):
        for q in self:
            q.put(item)
    
    def __getitem__(self, index):
        self.q_agg[index]

    def __iter__(self):
        if isinstance(self.q_agg, dict):
            return iter(self.q_agg.values())
        else:
            return iter(self.q_agg)
        


class ExpertPipelineStep(ABC):
    def __init__(self, name: str, args=tuple(), kwargs=dict(), is_daemon: bool = False):
        self.args = args
        self.kwargs = kwargs
        self.name = name
        self.is_daemon = is_daemon
        
        # placeholders for manager data
        self.mgr = None
        self.pipeline = None
        self.logger = None

        self.incoming_queue = Queue()

        self.__cur_p = None
        self._exit_flag = False

    def link_manager(self, mgr):
        self.mgr = mgr
        self.pipeline = mgr.pipeline
        self.logger = mgr.logger
        self.api = mgr.api

    @abstractmethod
    def run(self, q_in: Queue, q_out: AggQueue, *args, **kwargs):
        pass

    def start(self, q_in, q_out):
        args = (q_in, q_out) + self.args
        self.__cur_p = Thread(target=self.run, args=args, kwargs=self.kwargs,
                              name=self.name, daemon=self.is_daemon)
        self.__cur_p.start()

    def is_alive(self):
        if self.__cur_p:
            return self.__cur_p.is_alive()
        else:
            return False

    def exit_flag_up(self):
        self._exit_flag = True

    def __repr__(self) -> str:
        return self.name


class ExpertPipeline:
    STOP_MSG = 'STOP'

    def __init__(self, steps: List[Tuple[ExpertPipelineStep, ExpertPipelineStep]]):
        all_steps = set()
        self.incoming_queues = {}
        self.outgoing_queues = {}
        self.incoming_steps = {}
        self.outgoing_steps = defaultdict(dict)

        # rearange steps and step relationships
        for step_in, step_out in steps:
            all_steps.add(step_in)
            all_steps.add(step_out)
            self.incoming_steps.setdefault(step_out.name, set()).add(step_in)
            self.outgoing_steps.setdefault(step_in.name, set()).add(step_out)

        # save steps as list
        self.steps = sorted(all_steps, key=lambda s: s.name)

        # one incoming queue for each step
        for step in self.steps:
            self.incoming_queues[step.name] = step.incoming_queue
        
        # connect out step in queues as output for the prvious step
        for step in self.steps:
            self.outgoing_queues[step.name] = {}
            for out_step in self.outgoing_steps[step.name]:
                self.outgoing_queues[step.name][out_step.name] = self.incoming_queues[out_step.name]

            # make queues into AggQueue object for easy `batch-put`
            self.outgoing_queues[step.name] = AggQueue(self.outgoing_queues[step.name])

    def link_manager_to_steps(self, mgr):
        for step in self.steps:
            step.link_manager(mgr)

    def is_alive(self):
        return all(step.is_alive() for step in self.steps)

    def run(self):
        for step in self.steps:
            step.start(q_in=self.incoming_queues[step.name], q_out=self.outgoing_queues[step.name])

    def exit_all(self):
        for step in self.steps:
            step.exit_flag_up()
            step.incoming_queue.put(self.STOP_MSG)


def CLI_command(func):
    func._CLI_command = True
    return func

class global_config(ABC):
    def __init__(self, default_value) -> None:
        self._default_value = default_value
        self._value = default_value

    def get(self, msg: dict = None):
        if msg is not None:
            return msg.get(self.__class__.__name__, self._default_value)
        else:
            return self._value

    @abstractmethod
    def set(self, new_value: str):
        pass

    def from_msg(self):
        return 

class ExpertManager(ABC):
    def __init__(self, args: Namespace):
        self.args = args
        self.logger = self.__setup_logger(args.log)
        self.__set_global_configs()
        self.initialize()
        self.pipeline = self.get_pipeline()
        self.pipeline.link_manager_to_steps(self)

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_pipeline(self) -> ExpertPipeline:
        pass

    class output_dir(global_config):
        def __init__(self, default_value='./annotations'):
            super().__init__(default_value)

        def set(self, new_value: str):
            os.makedirs(new_value, exist_ok=True)
            self._value = new_value
    
    class output_style(global_config):
        def __init__(self, default_value=[OUTPUT_STYLE_JSON, OUTPUT_STYLE_ARANGO]):
            super().__init__(default_value)

        def set(self, new_value: str):
            values = new_value.split(',')
            bad_values = []
            for val in values:
                if val not in [OUTPUT_STYLE_ARANGO, OUTPUT_STYLE_ANNO, OUTPUT_STYLE_JSON]:
                    bad_values.append(val)
            
            if bad_values:
                raise ValueError(f'bad output style given: {bad_values}')
            
            self._value = values

    def run(self, timout=5):
        # run all pipeline threads
        self.pipeline.run()
        
        # wait for alive
        retries = 0
        while not self.pipeline.is_alive():
            if retries > timout:
                self.exit()
                raise TimeoutError('Pipeline did not boot in time')
            time.sleep(1)
            retries += 1

        # start CLI
        self.CLI()

    def print_n_log(self, s):
        print(s)
        self.logger.info(s)

    def CLI(self):
        print('infrastructure ready\n\n>>>', end=' ')
        self.logger.info('CLI ready to receive commands')
        sys.stdout.flush()

        supported_commands = self.get_cli_commands_dict()

        # iterate stdin commands
        for line in sys.stdin:
            # clean and log command
            line = line.strip()
            if line:
                self.logger.info(f'got CLI command: "{line}"')

            matching_commands = [cmd_name for cmd_name in supported_commands
                                 if line == cmd_name or line.startswith(f'{cmd_name} ')]
            
            # === commands switch start ===

            if not line:
                pass

            elif matching_commands:
                if len(matching_commands) > 1:
                    self.logger.error(f'ambiguous command "{line}". maching commands: {matching_commands}')
                else:
                    cmd_name = matching_commands[0]
                    self.logger.info(f'found matching command "{cmd_name}"')
                    cmd = supported_commands[cmd_name]
                    args_line = line[len(cmd_name):].strip()
                    
                    try:
                        terminate = cmd(self, args_line)

                        if terminate:
                            break
                    except:
                        self.logger.exception(f'could not execute command "{line}"')

            else:
                self.print_n_log(f'unsupported command: "{line}"')

            # === commands switch end ===

            print('>>>', end=' ')
            sys.stdout.flush()

    @CLI_command
    def status(self, line=''):
        """View running status of pipeline steps"""
        # log and print is_alive
        for step in self.pipeline.steps:
            out = f'{step.name}: {"alive" if step.is_alive() else "down"}'
            print(out)
            self.logger.info(out)

    @CLI_command
    def set(self, line=''):
        """
        Set a configuration: "set <cfg_name>=<value>" where cfg_name is one of the configurations.
        run "cfg" command to see possible configurations.
        """
        split_line = line.split('=')
        if len(split_line) != 2:
            raise ValueError(f'bad "set" command: {line}')

        name, value = split_line
        cfg = getattr(self, name, None)
        if cfg is not None and isinstance(cfg, global_config):
            cfg.set(value)
            self.logger.info(f'configuration set successfully: {name}={cfg.get()}')
        else:
            raise KeyError(f'"{name}" is not a valid configuration name')

    @CLI_command
    def cfg(self, line=''):
        """list all editable configurations"""
        for name, attr in self.get_current_config().items():
            self.print_n_log(f'{name}: {attr}')

    @CLI_command
    def commands(self, line=''):
        """list all available commands"""
        for cmd_name, cmd_func in self.get_cli_commands_dict().items():
            doc = cmd_func.__doc__
            self.print_n_log(f'{cmd_name}: {doc.strip() if doc else "?"}')

    @CLI_command
    def exit(self, line=''):
        """terminate all steps and quit the program"""
        self.print_n_log('closing pipeline and terminating CLI')
        self.pipeline.exit_all()
        return True  # signal termination

    @classmethod
    def get_cli_commands_dict(cls):
        cls_attrs = ((name, getattr(cls, name, None)) for name in dir(cls))
        return {name: getattr(cls, name) for name, attr in cls_attrs
                if callable(attr) and getattr(attr, '_CLI_command', False)}

    def get_current_config(self):
        return {name: attr.get() for name, attr in self.__dict__.items()
                if isinstance(attr, global_config)}

    def __setup_logger(self, log_file):
        """
        configure the global logger:
        - write DEBUG+ level to given `log_file`
        - write ERROR+ level to stderr
        - format: [time][thread name][log level]: message
        @param log_file: the file to which we wish to write. if an existing dir is given, log to a file
                        labeled with the curent date and time. if None, use the current working directory.
        """
        # create a logger for this instance
        logger = logging.getLogger(f'{self.__class__.__name__}-{log_file}')

        # set general logging level to debug
        logger.setLevel(logging.DEBUG)

        # choose logging format
        formatter = logging.Formatter('[%(asctime)s][%(threadName)s][%(levelname)s]: %(message)s')

        # create and add file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # create stderr stream handler
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.ERROR)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        title = f'===== {self.__class__.__name__} ====='
        logger.info('=' * len(title))
        logger.info(title)
        logger.info('=' * len(title))

        return logger

    def __set_global_configs(self):
        cfg_classes = [cls_attribute for _, cls_attribute in inspect.getmembers(self)
                       if inspect.isclass(cls_attribute)
                       and issubclass(cls_attribute, global_config)]

        arg_vars = vars(self.args)
        for cfg in cfg_classes:
            if cfg.__name__ in arg_vars:
                cfg_instance = cfg(arg_vars[cfg.__name__])
            else:
                cfg_instance = cfg()
            setattr(self, cfg.__name__, cfg_instance)

        self.output_dir.set(self.output_dir.get())
