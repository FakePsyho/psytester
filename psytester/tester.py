#!/usr/bin/env python3

# Author: Psyho
# Twitter: https://twitter.com/fakepsyho

#TODO:
# HIGH PRIORITY:
# -create proper ReadMe
# -fix double printing progress bug (get rid off \r?)
# -mode show simple histogram for stats
# -add simple automated unit tests (load config, run tests and check if output is as intended)
# -rename github/pypi to psytester
# -convert config to toml
# -better handling of crashing / infiniteloop programs
# -add crashes reporting (requires additional output from "cmd_play_game")
# -code is a bit messy now, refactor modes into functions
# -boostrapping to show % that particular solution is the best one (it's going to be very slow in python :())
# -change config file to toml

# LOW PRIORITY:
# -add a future proof mechanism for missing lines in config files? (will happen if someones updates the tester but the config file will stay the same)
# -add option to shorten group names (i.e. aliases)?
# -add support for custom scoring (cfg would be python code?)
# -sync with RUNNER? (probably to make RUNNER highly configurable instead)
# -add cleanup on ctrl+c (what that would be?)
# -simplify parameters in config (some parameters are redundant)

# ???:
# -show: add transpose (would be useful if someone wants to group by seed)?
# -is it possible to monitor cpu% and issue warning (too many threads); useful for running on cloud with tons of threads
# -add html export option for --show?
# -add comments to code (explaining functions should be enough)
# -add more annotations to functions
# -add ML model to figure out best parameters for test type? (this would require embedding prediction model in C++ code, simple MLP model? how to avoid overfitting?)


__version__ = '0.5.2'

import tabulate
import numbers
import re
import math
import sys
import os
import argparse
import subprocess
import glob
import json
import time
import configparser
import shutil
import traceback
import colorama
from typing import List, Dict, Union
import queue
from threading import Thread

global_time = time.time()

args = None
cfg = None
patterns = []

DEFAULT_CONFIG_PATH = 'tester.cfg'

tests_queue = queue.Queue()
results_queue = queue.Queue()


def try_str_to_numeric(x):
    if x is None:
        return None

    try: 
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x
            

def fatal_error(msgs, exit_main=False):
    if isinstance(msgs, str):
        msgs = [msgs]
    print('Fatal Error:', msgs[0])
    for msg in msgs[1:]:
        print(msg)
        
    if exit_main:
        os._exit(1)
    else:
        sys.exit(1)


def parse_color(color):
    color = color.upper()
    style = colorama.Style.NORMAL
    if '_' in color:
        style, color = color.split('_')
        style = getattr(colorama.Style, style)
    color = getattr(colorama.Fore, color) if color not in ['DEFAULT', '*'] else ''
    return style + color    


def run_test(test) -> Dict:
    retries_left = args.retry

    while True: 

        seed = test['seed']

        run_dir = args.name or '_default'
        
        output_dir = cfg["general"]["tests_dir"] + (f'/{run_dir}' if cfg["general"]["merge_output_dirs"].lower() == "false" else '')
        if args.debug:
            print()
            print('Running test:', test)
            print('Working directory:', output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        def parse_cmd(s):
            s = s.replace('%SEED%', str(seed))
            for i in range(1, 10):
                s = s.replace(f'%SEED0{i}%', f'{seed:0{i}}')
            s = s.replace('%OUTPUT_DIR%', output_dir)
            s = s.replace('%TESTER_ARGS%', args.tester_arguments)
            return s
        
        cmd = parse_cmd(cfg['general']['cmd_tester'])
        output_files = [parse_cmd(s) for s in cfg['general']['output_files'].split(',')]
        if args.debug:
            print('general/cmd_tester')
            print('Original:', cfg["general"]["cmd_tester"])
            print('Parsed:', cmd)
            print('general/output_files')
            print('Original:', cfg['general']['output_files'])
            print('Parsed:', output_files)
            print()
            print('Executing cmd_tester')
        
        
        start_time = time.time()
        completed_run = subprocess.run(cmd, shell=True)
        rv = {'id': seed}
        
        if args.debug:
            print(f'cmd_tester finished with {completed_run.returncode} return code after {round(time.time() - start_time, 6)} seconds')
            print()
            print('Contents of the output files:')
                
        
        for output_file in output_files:
            if args.debug:
                print(f'File: {output_file}')
                if not os.path.exists(output_file):
                    fatal_error(f'Output file {output_file} doesn\'t exist; this means either cmd_tester/output_files is incorrectly defined or your tester/solver crashed and didn\'t produce that file', True)
            with open(output_file) as f:
                for line in f:
                    for pattern in patterns:
                        m = re.match(pattern, line)
                        if m: 
                            if args.debug:
                                print(f'!!! Successful match with {pattern} pattern')
                            m = m.groupdict()
                            if 'VARIABLE' in m and 'VALUE' in m:
                                m[m['VARIABLE']] = m['VALUE']
                                del m['VARIABLE']
                                del m['VALUE']
                            rv.update({k: try_str_to_numeric(v) for k, v in m.items()})
                            if args.debug:
                                print('!!! Extracted dict:', m, 'New dict:', rv)
                        
            if cfg['general']['keep_output_files'].lower() != 'true':
                os.remove(output_file)
                    
        if 'score' not in rv:
            if retries_left:
                retries_left -= 1
                print(f'\r[Warning] Seed: {seed} cointains no score, retrying ({retries_left} retries left)')
                continue
            print(f'\r[Error] Seed: {seed} cointains no score')
            
        return rv
    
    
def find_res_files(dir='.', include_patterns=None, exclude_patterns=None):
    files_list = [path for path in os.listdir(dir) if path.endswith(cfg["general"]["results_ext"])]
    
    try:
        if include_patterns:
            files_set = set()
            for pattern in include_patterns:
                m = re.compile(pattern)
                for file in files_list:
                    if m.search(file):
                        files_set.add(file)
            files_list = list(files_set)
        if exclude_patterns:
            for pattern in exclude_patterns:
                m = re.compile(pattern)
                files_list = [file for file in files_list if not m.search(file)]
    except re.error:
        fatal_error('Supplied patterns are not valid regexp')
        
    return [f'{dir}/{file}' for file in files_list]
    
    
def load_res_file(path) -> Dict[int, float]:
    if not os.path.exists(path):
        fatal_error(f'Cannot locate {path} results file')

    with open(path) as f:
        lines = f.read().splitlines()
    results = [json.loads(line) for line in lines]
    return {result['id']: result for result in results} 


def process_raw_scores(scores: List[float], scoring: str) -> List[float]: 
    if scoring=='raw':
        return scores
    if scoring=='min':
        best_score = min([math.inf] + [score for score in scores if score > 0])
        return [best_score / score if score > 0 else 0 for score in scores]
    if scoring=='max':
        best_score = max([0] + [score for score in scores if score > 0])
        return [score / best_score if score > 0 else 0 for score in scores]
    
    fatal_error(f'Unknown scoring function: {scoring}')
        
    
def apply_filter(tests, data, filter):
    var, range = filter.split('=')
    if '-' in range:
        lo, hi = range.split('-')
        lo = try_str_to_numeric(lo) if lo else min([data[test][var] for test in tests])
        hi = try_str_to_numeric(hi) if hi else max([data[test][var] for test in tests])
        return [test for test in tests if lo <= data[test][var] <= hi]
    else:
        value = try_str_to_numeric(range)
        return [test for test in tests if data[test][var] == value]

    
def show_summary(runs: Dict[str, Dict[int, float]], tests: Union[None, List[int]] = None, data=None, groups=None, filters=None):
    if not tests:
        tests_used = [set(run_results.keys()) for run_name, run_results in runs.items()]
        tests = tests_used[0].intersection(*tests_used[1:])
    else:
        # TODO: error check if tests are cointained in intersection of all results files?
        pass

    if not tests:
        fatal_error('There are no common tests within the results files (maybe one of the results files is empty?)')

    if not data and (filters or groups):
        fatal_error('Filters/Groups used but no data file is provided')

    if filters:
        initial_tests_no = len(tests)
        for filter in filters:
            tests = apply_filter(tests, data, filter)
        print(f'Filtered {initial_tests_no} tests to {len(tests)}')
            
    # init color commands
    even_row_cmd = ''
    odd_row_cmd = ''
    header_cmd = ''
    group_max_cmd = ''
    reset_cmd = ''
    if cfg['general']['show_colors'].lower() == 'true':
        try:
            colorama.init()
            even_row_cmd = parse_color(cfg['general']['even_row_color'])
            odd_row_cmd  = parse_color(cfg['general']['odd_row_color'])
            header_cmd = parse_color(cfg['general']['header_color'])
            group_max_cmd = parse_color(cfg['general']['group_max_color'])
            reset_cmd = colorama.Style.RESET_ALL
        except AttributeError:
            traceback.print_exc()
            fatal_error('One of the specified colors is not valid. Refer to the cfg file for valid combinations')


    # create groups
    group_names = ['Overall']
    group_tests = [tests]
    
    if groups:
        for group in groups:
            var = None
            if '=' in group: var = group.split('=')[0]
            if '@' in group: var = group.split('@')[0]
            if '=' not in group and '@' not in group: var = group
            var_missing = [var not in data[test] for test in tests]
            if all(var_missing):
                fatal_error([f'Variable {var} doesn\'t exist in the data file', f'Only the following variables are present: {set().union(*[set(data[test].keys()) for test in tests])}'])
            if any(var_missing):
                fatal_error(f'Variable {var} is missing from {sum(var_missing)} out of {len(tests)} tests')
            
            if '=' in group:
                group_names.append(group)
                group_tests.append(apply_filter(tests, data, group))
            elif '@' in group:
                bins = int(group.split('@')[1])
                values = sorted([data[test][var] for test in tests])
                # XXX: probably there's a better way to split values into bins
                pos_start = 0
                for bin in range(bins):
                    pos_end = (bin+1) * len(tests) // bins
                    while pos_end < len(tests) and values[pos_end] == values[pos_end-1]: pos_end += 1
                    if pos_end <= pos_start: 
                        continue
                    group_name = f'{var}={values[pos_start]}-{values[pos_end-1]}'
                    group_names.append(group_name)
                    group_tests.append(apply_filter(tests, data, group_name))
                    pos_start = pos_end
            else:
                var_set = sorted(set([data[test][var] for test in tests]))
                for value in var_set:
                    group_names.append(f'{var}={value}')
                    group_tests.append(apply_filter(tests, data, f'{var}={value}'))
                    
    # generate data for each column
    columns = {}
    columns['runs'] = [('Tests\nRun', [run_name for run_name in runs])]
    columns['groups'] = []

    max_cells = []

    precision = int(cfg["general"]["precision"])
        
    # TODO: speed up (find the bottleneck, maybe try numpy?)

    total_fails = {run_name: 0 for run_name in runs}
    total_bests = {run_name: 0 for run_name in runs}
    total_uniques = {run_name: 0 for run_name in runs}
    total_gain = {run_name: 0 for run_name in runs}
    total_missing = {run_name: 0 for run_name in runs}
    for group_no, (group_name, group_test) in enumerate(zip(group_names, group_tests)):
        total_scores = {run_name: 0 for run_name in runs}
        group_scale = args.scale / max(1, len(group_test)) if args.scale else 1.0
        for test in group_test:
            scores = process_raw_scores([run_results[test].get(args.var, 0) for run_results in runs.values()], args.scoring)
            best_score = max(scores)
            second_best_score = sorted(scores)[-2] if len(scores) > 1 else 0
            unique_best = len([score for score in scores if score == best_score]) == 1
            for run_name, score in zip(runs.keys(), scores):
                total_scores[run_name] += score
                if group_no == 0:
                    total_bests[run_name] += 1 if score == best_score else 0
                    total_uniques[run_name] += 1 if score == best_score and unique_best else 0
                    total_gain[run_name] += max(0, score - second_best_score) * group_scale
                    total_fails[run_name] += 1 if score <= 0 else 0
                    total_missing[run_name] += 0 if args.var in runs[run_name][test] else 1

                    
        column = (f'{len(group_test)}\n{group_name}', [total_scores[run_name] * group_scale for run_name in runs])

        # mark the position of the best score in the group
        best_score = max(total_scores.values())
        for i, score in enumerate(total_scores.items()):
            if score[1] == best_score:
                max_cells.append((column[0], i))
                # column[1][i] = group_max_cmd + str(column[1][i]) + reset_cmd + (even_row_cmd if i % 2 else odd_row_cmd)

        if group_no == 0:
            columns['overall'] = [column]
        else:
            columns['groups'].append(column)
     
    columns['bests'] = [('\nBests', [total_bests[run_name] for run_name in runs])]
    columns['uniques'] = [('\nUniques', [total_uniques[run_name] for run_name in runs])]
    columns['gain'] = [('\nGain', [total_gain[run_name] for run_name in runs])]
    columns['fails'] = [('\nFails', [total_fails[run_name] for run_name in runs])]
    columns['missing'] = [('\nMissing', [total_missing[run_name] for run_name in runs])]
    
    if all([v > 0 for v in total_missing.values()]):
        fatal_error(f'None of the results files contain "{args.var}" variable')

    leaderboard = cfg['general']['leaderboard_score'] if args.var == 'score' else cfg['general']['leaderboard_custom']
    leaderboard = 'runs,' + leaderboard
    headers = []
    table = []

    # TODO: show error if it's not avg,max,min,sum?
    # TODO: add an ability to add alias with = so the final format would FUN:VAR.X=ALIAS
    # TODO: rewrite this part since it's a complete mess

    # generate var columns
    all_vars = [(column_name[:3].lower(), column_name[4:].split('.')[0]) for column_name in leaderboard.split(',') if column_name.lower()[:4] in ['avg:','min:','max:','sum:']]
    for fun, var in all_vars:
        column = []
        for run_results in runs.values():
            if not any([var in run_results[test] for test in tests]):
                column.append(None)
                continue
            data = [run_results[test][var] for test in tests if var in run_results[test]]
            fun_mapping = {'sum': sum, 'min': min, 'max': max, 'avg': lambda arr: sum(arr) / len(tests)}
            column.append(fun_mapping[fun](data))
        columns[f'{fun}:{var}'] = [(f'\n{var}', column)]

    for column_name in leaderboard.split(','):
        column_name = column_name.lower() if not column_name.lower().startswith('var:') else 'var:' + column_name[4:]
        optional = False
        column_precision = precision
        if column_name[-1] == '?':
            optional = True
            column_name = column_name[:-1]
        if column_name[-2] == '.' and column_name[-1].isdigit():
            column_precision = int(column_name[-1])
            column_name = column_name[:-2]
        if column_name not in columns:
            fatal_error(f'Unknown column name: {column_name}, please correct the leaderboard_XXX option')
        for column in columns[column_name]:
            if not optional or any(column[1]):
                data = column[1]
                data = [round(value, column_precision if column_precision > 0 else None) if value is not None and isinstance(value, numbers.Number) else value for value in data]
                for col_header, row in max_cells:
                    if col_header == column[0]:
                        data[row] = group_max_cmd + str(data[row]) + reset_cmd + (even_row_cmd if row % 2 else odd_row_cmd)
                headers.append(column[0])
                table += [data]
                
    table = list(zip(*table))
    if hasattr(tabulate, 'MIN_PADDING'):
        tabulate.MIN_PADDING = 0    

    # generate ascii table and output it using colors
    output = tabulate.tabulate(table, headers=headers)
    parity = False
    header = True
    for line in output.splitlines():
        if header:
            if line.startswith('-'):
                header = False
            else:
                line = header_cmd + line + reset_cmd
        else:
            line = (even_row_cmd if parity else odd_row_cmd) + line + reset_cmd
            parity = not parity
        print(line)
    print(reset_cmd, end='')


def _main():
    global args
    global cfg
    
    parser = argparse.ArgumentParser(description='Local tester for Topcoder Marathons & AtCoder Heuristic Contests\nMore help available at https://github.com/FakePsyho/psytester', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default=DEFAULT_CONFIG_PATH, help='path to cfg file')
    parser.set_defaults(mode=None)
    subparsers = parser.add_subparsers(title='modes')
    
    parser_args_tests = argparse.ArgumentParser(add_help=False)
    parser_args_tests.add_argument('-t', '--tests', type=str, help='number of tests to run (seeds 1-N), range of seeds (e.g. A-B) or the name of the JSON/text file with the list of seeds')
    
    parser_args_data = argparse.ArgumentParser(add_help=False)
    parser_args_data.add_argument('--data', type=str, default=None, help='file with metadata, used for grouping and filtering; in order to always use latest results file set it to LATEST') 
    parser_args_data.add_argument('--filters', type=str, default=None, nargs='+', metavar='FILTER', help='filters results based on the provided criteria in the form of VAR=A-B') 
    parser_args_data.add_argument('--dir', type=str, default=None, help='directory where the results files are located')

    parser_run = subparsers.add_parser('run', aliases=['r'], parents=[parser_args_tests], help='runs a batch of tests')
    parser_run.set_defaults(mode='run')
    parser_run.add_argument('-m', '--threads_no', type=int, help='number of threads to use') 
    parser_run.add_argument('-p', '--progress', action='store_true', help='shows current progress when testing') 
    parser_run.add_argument('-d', '--debug', action='store_true', help='special verbose mode helpful when figuring out why psytester doesn\'t work; overrides threads_no to 1')
    parser_run.add_argument('-a', '--tester_arguments', type=str, default='', help='additional arguments for the tester')
    parser_run.add_argument('-r', '--retry', type=int, default=0, help='number of retries when a test fails')
    parser_run.add_argument('name', type=str, nargs='?', default=None, help='name of the run; if not specified the results will be printed to stdout') 
    
    parser_show = subparsers.add_parser('show', aliases=['s'], parents=[parser_args_tests, parser_args_data], help='shows current results', formatter_class=argparse.RawTextHelpFormatter)
    parser_show.set_defaults(mode='show')
    parser_show.add_argument('--groups', type=str, nargs='+', default=None, metavar='GROUP', help='create additional columns for each group based on the provided criteria\nFormats allowed:\nVAR     - create a single group for each value of VAR\nVAR@N   - create N equal-sized based on VAR\nVar=A-B - create a single group where VAR is within A-B range') 
    parser_show.add_argument('--var', type=str, default='score', help='name of the variable you want to visualize (instead of score)')
    parser_show.add_argument('--scoring', type=str, default=None, choices=['raw','min', 'max'], help='sets the scoring function used for calculating ranking')
    parser_show.add_argument('--sorting', type=str, default=None, choices=['name', 'date'], help='sets how the runs are sorted')
    parser_show.add_argument('--files', type=str, nargs='+', default=None, help='list of regexp patterns for files to be included')
    parser_show.add_argument('--xfiles', type=str, nargs='+', default=None, help='list of regexp patterns for files to be excluded')
    parser_show.add_argument('--scale', type=float, default=None, help='sets scaling of results') 
    parser_show.add_argument('--noscale', action='store_true', help='turns off the scaling; values will be a simple sum over all tests') 
    
    parser_find = subparsers.add_parser('find', aliases=['f'],  parents=[parser_args_tests, parser_args_data], help='sorts the results based on specified critieria')
    parser_find.set_defaults(mode='find')
    parser_find.add_argument('--var', type=str, default='score', help='name of the variable you want to sort by')
    parser_find.add_argument('--order', type=str, default='-', choices=['-', '+'], help='whether the tests should be sorted descending ("-") or ascending ("+")')
    parser_find.add_argument('--limit', type=int, default=None, help='limits the number of tests to print') 
    
    parser_config = subparsers.add_parser('config', aliases=['c'], help='loads/saves/deletes specified template config')
    parser_config.set_defaults(mode='config')
    config_mode_group = parser_config.add_mutually_exclusive_group(required=True)
    config_mode_group.add_argument('--load', type=str, metavar='TEMPLATE', help='creates a new config based on specified template config')
    config_mode_group.add_argument('--save', type=str, metavar='TEMPLATE', help='updates a template config with local config')
    config_mode_group.add_argument('--delete', type=str, metavar='TEMPLATE', help='permanently deletes stored template config')
    config_mode_group.add_argument('--list', action='store_true', help='lists available template configs')
    
    args = parser.parse_args()

    if args.mode is None:
        fatal_error('No mode specified, type "psytester -h" for help')
    
    if args.mode == 'config':
        if args.load:
            template = args.load + '.cfg'
            template_config = os.path.join(os.path.dirname(__file__), template)
            if not os.path.exists(template_config):
                fatal_error(f'Missing {template} template config file')
            if os.path.exists(args.config):
                fatal_error(f'Config file {args.config} already exists')
            print(f'Creating new config file at {args.config}')
            shutil.copy(template_config, os.path.join(os.getcwd(), args.config))
            sys.exit(0)
            
        elif args.save:
            template = args.save + '.cfg'
            template_config = os.path.join(os.path.dirname(__file__), template)
            if os.path.exists(template_config):
                fatal_error(f'Template config file {args.save} already exists; if you wish to update it, you have to delete it first')
            if not os.path.exists(args.config):
                fatal_error(f'Config file {args.config} doesn\'t exist')
            print(f'Updating {args.save} template config with {args.config}')
            shutil.copy(os.path.join(os.getcwd(), args.config), template_config)
            sys.exit(0)
            
        elif args.delete:
            template = args.delete + '.cfg'
            template_config = os.path.join(os.path.dirname(__file__), template)
            if not os.path.exists(template_config):
                fatal_error(f'Missing {template} template config file')
            print(f'Removing template config file {template}')
            os.remove(template_config)
            sys.exit(0)
            
        elif args.list:
            template_configs = glob.glob(f'{os.path.dirname(__file__)}/*.cfg')
            table = []
            for template_config in template_configs:
                cfg = configparser.ConfigParser()
                cfg.read(template_config)
                table += [[os.path.splitext(os.path.basename(template_config))[0], cfg['general']['description']]]
            print('Available template config files:')
            print(tabulate.tabulate(table, headers=['name', 'description']))
            sys.exit(0)
            
        
    if not os.path.exists(args.config):
        fatal_error([f"Missing config file {args.config}, either use correct config file with \"psytester -c config_file\" or create a new one with \"psytester config --load template\"",
            "If you don't know how to use psytester, please check out the github project readme at: https://github.com/FakePsyho/psytester"])
    
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(args.config)
    
    if cfg['general']['version'] != __version__:
        fatal_error([f"{args.config} version ({cfg['general']['version']}) doesn't match the current version of psytester {__version__}",
            "Unfortunately psytester is currently not backwards compatible with old config files",
            "The recommended way to resolve this problem is to manually update your config file with changes introduced in the new version (create a new config file with \"psytester config --load template\")",
            "Alternatively, you can downgrade your version of psytester to match the config file"])
    
    # XXX: probably there's a better way to do this
    def convert(value, type=str):
        if value is None or value == '':
            return None
        
        if type == bool:
            return value.lower() in ['true', 'yes']
        return type(value)
        
    
    # Parse args.tests
    args.tests = try_str_to_numeric(args.tests or convert(cfg['default']['tests']))
    if args.tests is None:
        pass
    elif isinstance(args.tests, int):
        args.tests = list(range(1, args.tests + 1))
    elif re.search('[a-zA-Z]', args.tests):
        if not os.path.exists(args.tests):
            fatal_error(f'Cannot locate {args.tests} file')

        with open(args.tests) as f:
            lines = f.read().splitlines()
        assert len(lines) > 0
        
        if isinstance(try_str_to_numeric(lines[0]), int):
            args.tests = [int(line) for line in lines]
        else:
            args.tests = [json.loads(line)['id'] for line in lines]
    else:
        assert '-' in args.tests
        lo, hi = args.tests.split('-')
        lo = try_str_to_numeric(lo)
        hi = try_str_to_numeric(hi)
        args.tests = list(range(lo, hi + 1))
    
    # Mode: Find
    if args.mode == 'find':
        args.data = args.data or cfg['default']['data']
        if args.data == 'LATEST':
            results_files = find_res_files(args.dir or cfg['general']['results_dir'])
            _, args.data = sorted(zip([os.path.getmtime(result_file) for result_file in results_files], results_files))[-1]
        else:
            args.data += cfg['general']['results_ext']
        
        results = load_res_file(args.data)
        tests = results.keys()
        if args.tests:
            tests = list(set(tests) & set(args.tests))
        
        for filter in args.filters or []:
            tests = apply_filter(tests, results, filter)
            
        ordered_tests = [test for _, test in sorted(zip([results[test][args.var] for test in tests], tests), reverse=args.order == '-')]
        
        if args.limit:
            ordered_tests = ordered_tests[:args.limit]
        
        print(f'Finding in {args.data} file')
        for test in ordered_tests:
            print(json.dumps(results[test]))
        sys.exit(0)
        
    # Mode: Show
    if args.mode == 'show':
        args.data = args.data or cfg['default']['data']
        args.scale = args.scale or convert(cfg['default']['scale'], float)
        if args.noscale:
            args.scale = None
        args.scoring = args.scoring or convert(cfg['default']['scoring'])
        args.sorting = args.sorting or convert(cfg['default']['sorting'])
        
        results_files = find_res_files(args.dir or cfg['general']['results_dir'], args.files, args.xfiles)
        if not results_files:
            fatal_error(f'There are no results files in the results folder: {cfg["general"]["results_dir"]}')
            
        if args.sorting == 'name':
            results_files = sorted(results_files)
        elif args.sorting == 'date':
            results_files = [result_file for _, result_file in sorted(zip([os.path.getmtime(result_file) for result_file in results_files], results_files))]
            
        results = {os.path.splitext(os.path.basename(file))[0]: load_res_file(file) for file in results_files}
        
        if args.data == 'LATEST':
            _, args.data = sorted(zip([os.path.getmtime(result_file) for result_file in results_files], results_files))[-1]
        data_file = load_res_file(args.data) if args.data and os.path.isfile(args.data) else None
        
        show_summary(results, tests=args.tests, data=data_file, groups=args.groups, filters=args.filters)
        sys.exit(0)

    # Mode: Run
    if args.mode == 'run':
        args.threads_no = args.threads_no or convert(cfg['default']['threads_no'], int)
        args.progress = args.progress or convert(cfg['default']['progress'], bool)
        args.tester_arguments = args.tester_arguments or cfg['default']['tester_arguments'] 
        if args.debug:
            args.threads_no = 1
            args.progress = False
        
        if not os.path.exists(cfg['general']['tests_dir']):
            os.mkdir(cfg['general']['tests_dir'])
            
        if not args.tests:
            fatal_error('You need to specify tests to run, use --tests option')
        
        assert args.threads_no >= 1
        fout = sys.stdout
        if args.name:
            os.makedirs(cfg["general"]["results_dir"], exist_ok=True)
            fout = open(f'{cfg["general"]["results_dir"]}/{args.name}{cfg["general"]["results_ext"]}', 'w')
            
        global patterns
        for s in cfg['general']:
            if (s.startswith('extraction_regex_')):
                patterns.append(cfg['general'][s])
        if not patterns:
            fatal_error('No extraction patterns specified (introduced in psytester 0.5.0) - check online documentation')
        if args.debug:
            print('Extraction patterns found:')
            for pattern in patterns:
                print(pattern)
            print()
        
        try:
            start_time = time.time()
            for id in args.tests:
                tests_queue.put({'seed': id})
            tests_left = args.tests
            
            def worker_loop():
                while True:
                    try:
                        seed = tests_queue.get(False)
                        result = run_test(seed)
                        results_queue.put(result)
                        if args.debug:
                            time.sleep(0.1)
                    except queue.Empty:
                        return
                    except:
                        traceback.print_exc()
                        fatal_error('One of the worker threads encountered an error', exit_main=True);
            
            workers = [Thread(target=worker_loop) for _ in range(args.threads_no)]
            for worker in workers:
                worker.start()
            
            sum_scores = 0
            log_scores = 0
            results = {}
            processed = 0
            
            while tests_left:
                result = results_queue.get()
                results[result['id']] = result
                assert result['id'] in tests_left
                while tests_left and tests_left[0] in results:
                    processed += 1
                    seed = tests_left[0]
                    print(json.dumps(results[seed]), file=fout, flush=True)
                    tests_left = tests_left[1:]
                    sum_scores += results[seed]['score'] if results[seed]['score'] > 0 else 0
                    log_scores += math.log(results[seed]['score']) if results[seed]['score'] > 0 else 0
                    if args.progress and args.name:
                        output = f'Progress: {processed} / {processed+len(tests_left)}   Time: {time.time() - start_time : .3f}'
                        print(f'\r{output}                       ', end='', file=sys.stderr)
                        sys.stderr.flush()
                        time.sleep(0.002)
        except KeyboardInterrupt:
            print('\nInterrupted by user', file=sys.stderr)
            os._exit(1)
            
        print(file=sys.stderr)
            
        print("Time:", time.time() - start_time, file=sys.stderr)
        print("Avg Score:", sum_scores / len(results), file=sys.stderr)
        print("Avg Log Scores:", log_scores / len(results), file=sys.stderr)
    
    
if __name__ == '__main__':
    _main()