# Pluggable experts

## Using an expert


### Install Environment
- Go to you desired expert's directory and see specific instructions (most likely some bash setup script).


### Command-line Options
- run `python run_expert --help` to see general options.
- run `python run_expert <expert_name> --help` to view expert specific options. E.g. `python run_expert tracker --help`.
- run `python run_expert <expert_name>` to run with default options.


### Command-line Interface
- you will be prompted for commands with the prompt `>>>`.
- sometimes we recieve warnings that hide the prompt from you. just hit `enter` a few times and you will see it again.
- available configurations can be seen using the `cfg` command and set using the `set` command.
- to change a configuration, e.g. `batch_size`, run the command: `set <config_name>=<legal_config_value>`. E.g. `set batch_size=5`.
- to see the running status of the pipeline, run the `status` command.
- available commands can be seen by running the `commands` command. commands may change between experts.
- currently we have `actions` and `tracker` experts, both with `local` command (local video) and `remote` command (remote video).


## Adding a New Expert
Using the experts is the easy part. To add a new expert, we need to do several things.

### Expert Directory
- An expert is identified by its directory name.
- All expert-specific code must go in this directory.

### Expert Package
- The expert should be enclosed within its own package within the directory. See `tracker.autotracker` and `actions.stepwrapper`.
- This package will provide an API for the expert manager (see below).
- It is recommended to leave a .ipynb file as a demo for how to use this package as the author intended. For this reason, it is good to add a custom annotator (see next section).

### Create Custom Utilities
- Annotator - create a subclass of `common.BaseAnnotator`. You are required to implement the abstract method `overlay_prediction_on_frame` which tells the annotator how to add a frame's matching annotations into the video itself. See `actions.ActionAnnotator` for an example.
- API utility -  create a subclass of `common.RemoteAPIUtility`. This class gives you access to S3 and arango. We extend this class to have contain the API saving the expert's outputs into the remote database. see `actions.ActionsAPIUtility` for an example.

### Custom Expert Manager
We now want to create the pipeline for this expert to work at scale. for this we have 3 helper classes from `common.ExpertManager`:
- ExpertPipeline
- ExpertPipelineStep
- ExpertManager


#### ExpertPipelineStep
This is basically a thread wrapper. For each thread you must create a separate step. This is done by extending `common.ExpertManager.ExpertPipelineStep`. You are required to implement the `run` function, which accepts as input the input queue and all the output queues. The output queues are a custom type called `AggQueue`, which can `put` into multiple queues at once. This will be all the input queues of the threads with which this step communicates. Once a step has been added to a pipeline, the pipeline's manager and all its most important fields (see `link_manager` funciton).

#### ExpertPipeline
This class will create the communication map (queues) for the multithreaded pipeline. You do not need to extend this class, only use it to build yor pipeline. This is done by defining the relationships between the different steps via a list of tuples. For example, if we would like step1 to communicate with step2, step2 to communicate with step3, and step4 to communicate with both steps 1 and 2, then we would initialize like so:

    ExpertPipeline([
        (step1, step2)
        (step2, step3),
        (step4, step1),
        (step4, step2)
    ])

We do this in the manager's `get_pipeline` funciton. See `tracker.TrackerManager.get_pipeline` for an example.


#### ExpertManager
This class handles the pipeline and the user interface for this expert. You must extend this class for your expert (see `actions.ActionsManager` for an example). Then you must implement the following (command line arguments are always available at `self.args`):
- The `initialize` function - after the logger has been created and the configurations have been initialize, you may perform some expert-specific initializations. 
- The `get_pipeline` function - this is where you define the expert's pipeline.
- CLI commands - these are defined as functions. you must implement commands as regular instance method functions in your ExpertManager subclass that accept a single argument, the command arguments string as typed by the user. To mark this method as a command, use the `@CLI_command` decorator above the function. You may thrwo exceptions and they will be caught by the superclass (see `common.ExpertManager.ExpertManger.CLI` function).
- Global Configurations - The `global_config` class is the superclass for defining configurations. configurations can do 2 things, get and set. the `get` function takes the current value of the configuration. For thread-safety purposes, `get` can accept a dictionary input that contains an old version of the configuration value (in case it is updated during the expert's run). We usually get this dictionary early using the `get_current_config` function. The `set` function is an abstract method that must be implemented for each configuration. This si because we want to so that they are saved as the correct type. For example, see the superclass configuration `output_style`, which can accept comma delimited values and saves them as a list.


### Link to Expert Runner
There is a `run_expert.py` script in the `experts` directory. This script parses the command line arguments and runs the correct expert manager. You must add your custom expert here:
- Add your expert to the `managers_dict` dictionary. The key must be the expet's name (directory name) and the value must be the name of the file where you implemented your custom ExpertManager. For this to work correctly, the name of the module and the manager itself must have the same name. E.g. the module `actions/ActionsManager.py` contains the class `ActionManager` that inherits from `ExpertManager`.
- in the `parse_args` function, add any custom command line arguments. you must first use the `subparsers` object to create a subparser for you expert. Note that its name must be the expert's name (directory name). It is recommended to handle default behavior in the your manager's `initialize` funciton.

After the above steps you will be able to run your expert as explained in the "Using an expert" section.
