import argparse
import os

from datetime import datetime
from importlib import import_module

from common.ExpertManager import OUTPUT_STYLE_ANNO, OUTPUT_STYLE_ARANGO, OUTPUT_STYLE_JSON, OUTPUT_DEFAULT


managers_dict = {
    # dirname: manager_namne (module + class)
    'tracker': 'TrackerManager',
    'actions': 'ActionsManager'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run ojbect detection and tracking engines. For more '
                                                 'detailed instructions see '
                                                 'NEBULA2/experts/instructions.txt')


    subparsers = parser.add_subparsers(dest='expert', help='Type of expert manager tool.')

    parser.add_argument('--log', '-l',
                        default=None,
                        help='path to output log file. If an existing directory is given, then the log '
                             'file name will be a default name "YYY-MM-dd_HH:mm:ss.log". If not provided, '
                             'the default filename is used in the current working directory.')
    parser.add_argument('--no-arango',
                            action='store_false',
                            dest='arango',
                            default=True,
                            help='Adding this flag disables the arango client scheduler daemon.')
    parser.add_argument('--output-style', '-o',
                        nargs="+",
                        default=None,
                        choices=[OUTPUT_STYLE_JSON, OUTPUT_STYLE_ANNO, OUTPUT_STYLE_ARANGO],
                        help='The method for saving the tracking output. If "json" or "anno" are set, '
                             'the output is redirected to the location of the --output-dir argument. The'
                             'default is "arango" if the --no-arango flag is NOT set, or "json" otherwise.')
    parser.add_argument('--output-dir', '-d',
                        default=OUTPUT_DEFAULT,
                        help='For use only when --output-style "json" or "anno" are set. Indicates wher to '
                             'save the annotations. If "json" or "anno" are set but no --output-dir is '
                             'provided, a directory called "annotations/" is created in the current working '
                             'directory.')
    # ====================================
    # = subparsers for different experts =
    # ====================================

    # ===== Tracker args =====
    tracker_parser = subparsers.add_parser("tracker")
    tracker_parser.add_argument('--backend', '-b',
                        default=None,
                        help='Detection model backend. If not provided, any available backend may be '
                             'used. If a model configuration is provided via the --model argument, then '
                             'a backend that has that configuration will be used, if one is available.',
                        choices=['tflow', 'detectron'])
    tracker_parser.add_argument('--model', '-m',
                        default=None,
                        help='Detection model configuration. Should be one of the CFG constants in the '
                             'chosen model backend. If not provided, a default configuration for the '
                             'backend is used.')
    tracker_parser.add_argument('--confidence', '-c',
                        default=0.6,
                        type=float,
                        help='The model prediction confidence threshold (default is 0.6).')
    

    # ===== actions args args =====
    # no extra arguments. this is here to add the "actions" option
    actions_parser = subparsers.add_parser("actions")



    # parse arguments
    parsed_args = parser.parse_args()
    parsed_args = __handle_args_defaults(parsed_args)

    return parsed_args


def __handle_args_defaults(parsed_args):
    """
    check parsed args validity and set default values if necessary.
    @param: parsed_args: a parsed arguments object from the command line.
    @return: the parsed arguments after default args set.
    """

    # === Logging Defaults ===
    # choose default logging location
    if not parsed_args.log:
        parsed_args.log = os.path.join('.', datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.log')
    elif os.path.isdir(parsed_args.log):
        parsed_args.log = os.path.join(parsed_args.log, datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.log')
    else:
        os.makedirs(os.path.dirname(parsed_args.log), exist_ok=True)

    # === Output Defaults ===
    # default output to arango without local save
    if not parsed_args.output_style:
        if parsed_args.arango:
            parsed_args.output_style = [OUTPUT_STYLE_ARANGO]
        else:
            parsed_args.output_style =  [OUTPUT_STYLE_JSON]

    # got local output save location. save as JSON
    elif not parsed_args.output_style:
        parsed_args.output_style = [OUTPUT_STYLE_JSON]
    
    # method arango with output file is incompatible
    elif parsed_args.output_style == [OUTPUT_STYLE_ARANGO]:
        print('Warning: output style "arango" does not save locally and will not save output to given path: {parsed_args.output_dir}')
    
    return parsed_args



if __name__ == "__main__":
    args = parse_args()
    print('running with config:')
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    mgr_name = managers_dict[args.expert]
    mgr_module = __import__(f'{args.expert}.{mgr_name}', fromlist=[mgr_name])
    mgr_class = getattr(mgr_module, mgr_name)

    mgr = mgr_class(args)

    mgr.run()
