#!/usr/bin/env python3

# Author: Psyho
# Twitter: https://twitter.com/fakepsyho

#TODO:
# HIGH PRIORITY:
# -create proper ReadMe
# -more error checking / clearer error messages
# -fix grouping/filtering if data file doesn't contain all test cases
# -config: merge subgroups (you need at least one?)
# -fix double printing progress bug
#  -above + customization of scripts in the config
# -mode show simple histogram for stats
# -find a way to make atcoder score consistent with local (score_mul parameter? / is it needed?)
# -add option to print parsed commands (or maybe just print when encountered an error?)
# -add an option for custom scoreboard ordering? (would simply show_XXX options)
# -exec: add verbose/debug option that prints a lot of additional stuff in order to make debugging easier
# -add simple automated unit tests (load config, run tests and check if output is as intended)

# LOW PRIORITY:
# -add a future proof mechanism for missing lines in config files? (will happen if someones updates the tester but the config file will stay the same)
# -add option to shorten group names?
# -use --tests for --find?
# -add support for custom scoring (cfg would be python code?)
# -add RUNNER parameters (like for hash code) (Moved to RUNNER?)
# -add batching? (Moved to RUNNER?)
# -sync with RUNNER? (how?)
# -add cleanup on ctrl+c (what that would be?)
# -change to subparsers (exec / show / find?)
# -simplify parameters in config (some parameters are redundant)
# -add autodetect for atcoder run/gen cmd (should be easy if files have original names)
# -add some lock against running atcoder's gen multiple times at the same time
# -improve script generation?

# ???:
# -show: add transpose?
# -is it possible to monitor cpu% and issue warning (too many threads); useful for running on cloud with tons of threads
# -add html export option for --show?
# -add comments to code (explaining functions should be enough)
# -add more annotations to functions

__version__ = '0.4.1'

import tabulate
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
import _thread
import regex
from typing import List, Dict, Union
import queue
from threading import Thread

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


def run_test(test) -> Dict:
    seed = test['seed']

    run_dir = args.name or '_default'
    
    output_dir = cfg["general"]["tests_dir"] + (f'/{run_dir}' if cfg["general"]["merge_output_dirs"].lower() == "false" else '')
    
    os.makedirs(output_dir, exist_ok=True)
    
    def parse_cmd(s):
        s = s.replace('%SEED%', str(seed))
        for i in range(1, 10):
            s = s.replace(f'%SEED0{i}%', f'{seed:0{i}}')
        s = s.replace('%OUTPUT_DIR%', output_dir)
        s = s.replace('%TESTER_ARGS%', args.tester_arguments)
        if '%GEN_INPUT%' in s:
            s = s.replace('%GEN_INPUT%', str(test['path']))
        return s
    
    cmd = parse_cmd(cfg['general']['cmd_tester'])
    output_files = [parse_cmd(s) for s in cfg['general']['output_files'].split(',')]
    
    subprocess.run(cmd, shell=True)
    rv = {'id': seed}
    
    for output_file in output_files:
        with open(output_file) as f:
            for line in f:
                for pattern in patterns:
                    m = re.match(pattern, line)
                    if m: 
                        m = m.groupdict()
                        if 'VARIABLE' in m and 'VALUE' in m:
                            m[m['VARIABLE']] = m['VALUE']
                            del m['VARIABLE']
                            del m['VALUE']
                        rv.update({k: try_str_to_numeric(v) for k, v in m.items()})
                    
        if cfg['general']['keep_output_files'].lower() != 'true':
            os.remove(output_file)
                
    if 'score' not in rv:
        print(f'\r[Error] Seed: {seed} cointains no score')
        
        
    return rv
    
    
def find_res_files(dir='.'):
    return glob.glob(f'{dir}/*{cfg["general"]["results_ext"]}')
    
    
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
        best_score = min([math.inf] + [score for score in scores if score >= 0])
        return [best_score / score if score >= 0 else 0 for score in scores]
    if scoring=='max':
        best_score = max([0] + [score for score in scores if score >= 0])
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
            
    group_names = []
    group_tests = []
    
    group_names.append('Score')
    group_tests.append(tests)
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
        
    headers = ['Tests\nRun'] + [f'{len(tests)}\n{name}' for name, tests in zip(group_names, group_tests)]
        
    table = [[run_name] for run_name in runs]
    
    total_fails = {run_name: 0 for run_name in runs}
    total_bests = {run_name: 0 for run_name in runs}
    total_uniques = {run_name: 0 for run_name in runs}
    total_gain = {run_name: 0 for run_name in runs}
    total_missing = {run_name: 0 for run_name in runs}
    for group_no, group_test in enumerate(group_tests):
        total_scores = {run_name: 0.0 for run_name in runs}
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
                    
        for i, run_name in enumerate(runs):
            table[i].append(total_scores[run_name] * group_scale)
            
    missing = {run_name: v > 0 for run_name, v in total_missing.items()}
    if all(missing.values()):
        fatal_error(f'None of the results files contain "{args.var}" variable')
    
    if args.var == 'score':
        if cfg['general']['show_bests'].lower() == 'true':
            headers.append('\nBests')
            for i, run_name in enumerate(runs):
                table[i].append(total_bests[run_name])
        if cfg['general']['show_uniques'].lower() == 'true':
            headers.append('\nUniques')
            for i, run_name in enumerate(runs):
                table[i].append(total_uniques[run_name])
        if cfg['general']['show_gain'].lower() == 'true':
            headers.append('\nGain')
            for i, run_name in enumerate(runs):
                table[i].append(total_gain[run_name])
        if cfg['general']['autohide_fails'].lower() == 'false' or max(total_fails.values()) > 0:
            headers.append('\nFails')
            for i, run_name in enumerate(runs):
                table[i].append(total_fails[run_name])
    
    if any(missing.values()):
        headers.append('\nMissing')
        for i, run_name in enumerate(runs):
            table[i].append(total_missing[run_name])
        
    if hasattr(tabulate, 'MIN_PADDING'):
        tabulate.MIN_PADDING = 0
    print(tabulate.tabulate(table, headers=headers, floatfmt=f'.{cfg["general"]["precision"]}f'))
        

def _main():
    global args
    global cfg
    
    parser = argparse.ArgumentParser(description='Local tester for Topcoder Marathons & AtCoder Heuristic Contests\nMore help available at https://github.com/FakePsyho/mmtester', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default=DEFAULT_CONFIG_PATH, help='path to cfg file')
    subparsers = parser.add_subparsers(title='modes')
    
    parser_args_tests = argparse.ArgumentParser(add_help=False)
    parser_args_tests.add_argument('-t', '--tests', type=str, help='number of tests to run (seeds 1-N), range of seeds (e.g. A-B) or the name of the JSON/text file with the list of seeds')
    
    parser_args_data = argparse.ArgumentParser(add_help=False)
    parser_args_data.add_argument('--data', type=str, default=None, help='file with metadata, used for grouping and filtering; in order to always use latest results file set it to LATEST') 
    parser_args_data.add_argument('--filters', type=str, default=None, nargs='+', metavar='FILTER', help='filters results based on the provided criteria in the form of VAR=A-B') 

    parser_run = subparsers.add_parser('run', aliases=['r'], parents=[parser_args_tests], help='runs a batch of tests')
    parser_run.set_defaults(mode='run')
    parser_run.add_argument('-m', '--threads_no', type=int, help='number of threads to use') 
    parser_run.add_argument('-p', '--progress', action='store_true', help='shows current progress when testing') 
    parser_run.add_argument('-b', '--benchmark', type=str, default=None, help='benchmark res file to test against')
    parser_run.add_argument('-a', '--tester_arguments', type=str, default='', help='additional arguments for the tester')
    parser_run.add_argument('name', type=str, nargs='?', default=None, help='name of the run; if not specified the results will be printed to stdout') 
    
    parser_show = subparsers.add_parser('show', aliases=['s'], parents=[parser_args_tests, parser_args_data], help='shows current results', formatter_class=argparse.RawTextHelpFormatter)
    parser_show.set_defaults(mode='show')
    parser_show.add_argument('--groups', type=str, default=None, nargs='+', metavar='GROUP', help='create additional columns for each group based on the provided criteria\nFormats allowed:\nVAR     - create a single group for each value of VAR\nVAR@N   - create N equal-sized based on VAR\nVar=A-B - create a single group where VAR is within A-B range') 
    parser_show.add_argument('--var', type=str, default='score', help='name of the variable you want to visualize (instead of score)')
    parser_show.add_argument('--scoring', type=str, default=None, choices=['raw','min', 'max'], help='sets the scoring function used for calculating ranking')
    parser_show.add_argument('--sorting', type=str, default=None, choices=['name', 'date'], help='sets how the runs are sorted')
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
    config_mode_group.add_argument('--load', dest='template', type=str, metavar='TEMPLATE', help='creates a new config based on specified template config')
    config_mode_group.add_argument('--save', dest='template', type=str, metavar='TEMPLATE', help='updates a template config with local config')
    config_mode_group.add_argument('--delete', dest='template', type=str, metavar='TEMPLATE', help='permanently deletes stored template config')
    config_mode_group.add_argument('--list', action='store_true', help='lists available template configs')
    
    # parser_generate = subparsers.add_parser('generate', aliases=['g'], help='')
    # parser_generate.add_argument('--generate-scripts', action='store_true', help='generates scripts defined in the config file')
    # parser_generate.add_argument('--ip', type=str, default=None, help='optional argument for --generate-scripts')
    # parser_generate.add_argument('--source', type=str, default=None, help='optional argument for --generate-scripts')

    
    args = parser.parse_args()
    
    if args.mode == 'config':
        if args.config_mode == 'load':
            args.template += '.cfg'
            template_config = os.path.join(os.path.dirname(__file__), args.template)
            if not os.path.exists(template_config):
                fatal_error(f'Missing {args.template} template config file')
            if os.path.exists(args.config):
                fatal_error(f'Config file {args.config} already exists')
            print(f'Creating new config file at {args.config}')
            shutil.copy(template_config, os.path.join(os.getcwd(), args.config))
            sys.exit(0)
            
        elif args.config_mode == 'save':
            args.template += '.cfg'
            template_config = os.path.join(os.path.dirname(__file__), args.template)
            assert os.path.exists(args.config)
            print(f'Updating {args.template} template config with {args.config}')
            # if os.path.exists(template_config):
                # print('Template config file {args.config_save} already exists, do you wish to overwrite it?')
            shutil.copy(os.path.join(os.getcwd(), args.config), template_config)
            sys.exit(0)
            
        elif args.config_mode == 'delete':
            args.template += '.cfg'
            template_config = os.path.join(os.path.dirname(__file__), args.template)
            if not os.path.exists(template_config):
                fatal_error(f'Missing {args.template} template config file')
            print(f'Removing template config file {args.template}')
            os.remove(template_config)
            sys.exit(0)
            
        elif args.config_mode == 'list':
            template_configs = glob.glob(f'{os.path.dirname(__file__)}/*.cfg')
            table = []
            for template_config in template_configs:
                cfg = configparser.ConfigParser()
                cfg.read(template_config)
                table += [[os.path.splitext(os.path.basename(template_config))[0], cfg['general']['description']]]
            print('Available template config files:')
            print(tabulate.tabulate(table, headers=['name', 'description']))
            sys.exit(0)
            
        else:
            assert false
        
    if not os.path.exists(args.config):
        fatal_error([f"Missing config file {args.config}, either use correct config file with \"mmtester -c config_file\" or create a new config file with \"mmtester --config-load config_template\"",
            "If you don't know how to use mmtester, please check out the github project readme at: https://github.com/FakePsyho/mmtester"])
    
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(args.config)
    
    if cfg['general']['version'] != __version__:
        fatal_error([f"{args.config} version ({cfg['general']['version']}) doesn't match the current version of mmtester {__version__}",
            "Unfortunately mmtester is currently not backwards compatible with old config files",
            "The easiest way to resolve the problem is to manually update your config file with changes introduced in the new version (create a new config file with --new-config)",
            "Alternatively, you can downgrade your version of mmtester to match the config file"])
    
    # XXX: probably there's a better way to do this
    def convert(value, type=str):
        if value is None or value == '':
            return None
        
        if type == bool:
            return value.lower() in ['true', 'yes']
        return type(value)
        
    
    # Mode: Generate Scripts (currently disable)
    # if args.mode == 'generate':
        # print('Generating Scripts')
        # for script_name in cfg['scripts']:
            # script = cfg.get('scripts', script_name, raw=True)
            # undefined = []
            
            # if '%RUN_CMD%' in script:
                # script = script.replace('%RUN_CMD%', cfg['general']['run_cmd'])
                
            # if '%EXEC%' in script:
                # if not args.exec:
                    # undefined.append('missing %ECEC% (use --exec EXEC)')
                # else:
                    # script = script.replace('%EXEC%', args.exec)
                    
            # if '%IP%' in script:
                # if not args.ip:
                    # undefined.append('missing %IP% (use --ip IP)')
                # else:
                    # script = script.replace('%IP%', args.ip)
                    
            # if '%SOURCE%' in script:
                # if not args.source:
                    # undefined.append('missing %SOURCE% (use --source SOURCE)')
                # else:
                    # script = script.replace('%SOURCE%', args.source)
                    
            # if undefined:
                # print(f'Ignoring script {script_name} because of {undefined}')
                # continue
                
            # with open(script_name, 'w') as f:
                # for i, line in enumerate(script.split('\\n')):
                    # prefix = f'{script_name} ='
                    # print(prefix if i == 0 else ' ' * len(prefix), line)
                    # print(line, file=f)
        # sys.exit(0)
    
    # Mode: Find
    if args.mode == 'find':
        args.data = args.data or cfg['default']['data']
        if args.data == 'LATEST':
            results_files = find_res_files(cfg['general']['results_dir'])
            _, args.data = sorted(zip([os.path.getmtime(result_file) for result_file in results_files], results_files))[-1]
        else:
            args.data += cfg['general']['results_ext']
        
        results = load_res_file(args.data)
        tests = results.keys()
        for filter in args.filters or []:
            tests = apply_filter(tests, results, filter)
            
        ordered_tests = [test for _, test in sorted(zip([results[test][args.var] for test in tests], tests), reverse=args.order == '-')]
        
        if args.limit:
            ordered_tests = ordered_tests[:args.limit]
        
        print(f'Finding in {args.data} file')
        for test in ordered_tests:
            print(json.dumps(results[test]))
        sys.exit(0)
        
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
    
    # Mode: Show
    if args.mode == 'show':
        args.data = args.data or cfg['default']['data']
        args.scale = args.scale or convert(cfg['default']['scale'], float)
        if args.noscale:
            args.scale = None
        args.scoring = args.scoring or convert(cfg['default']['scoring'])
        args.sorting = args.sorting or convert(cfg['default']['sorting'])
        
        results_files = find_res_files(cfg['general']['results_dir'])
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
        args.benchmark = args.benchmark or convert(cfg['default']['benchmark'])
        args.tester_arguments = args.tester_arguments or cfg['default']['tester_arguments'] 
        
        if not os.path.exists(cfg['general']['tests_dir']):
            os.mkdir(cfg['general']['tests_dir'])
            
        if not args.tests:
            fatal_error('You need to specify tests to run, use --tests option')
        
        assert args.threads_no >= 1
        fout = sys.stdout
        if args.name:
            os.makedirs(cfg["general"]["results_dir"], exist_ok=True)
            fout = open(f'{cfg["general"]["results_dir"]}/{args.name}{cfg["general"]["results_ext"]}', 'w')
            
        inputs_path = {}
        if cfg["general"]["cmd_generator"]:
            # generate input files 
            print('Generating test cases...', file=sys.stderr)
            gen_seeds = []
            if cfg['general']['generator_cache'].lower() == 'true':
                os.makedirs('inputs', exist_ok=True)
                inputs_path = {seed: f'inputs/{seed}.in' for seed in args.tests}
                present_inputs = set([path for path in os.listdir('inputs') if os.path.isfile(f'inputs/{path}')])
                gen_seeds = [seed for seed in args.tests if f'{seed}.in' not in present_inputs]
            else:
                inputs_path = {seed: f'in/{i:04d}.txt' for i, seed in enumerate(args.tests)}
                gen_seeds = args.tests
                
            if gen_seeds:
                seeds_path = 'mmtester_seeds.txt'
                with open(seeds_path, 'w') as f:
                    f.write('\n'.join([str(seed) for seed in gen_seeds]))
                subprocess.run(f'{cfg["general"]["cmd_generator"]} {seeds_path}', shell=True)
                if cfg['general']['generator_cache'].lower() == 'true':
                    for i, seed in enumerate(gen_seeds):
                        shutil.copy(f'in/{i:04d}.txt', f'inputs/{seed}.in')
            
        #TODO: add error handling/warning for benchmark file (file not existing, no full test coverage)
        benchmark = load_res_file(args.benchmark + cfg['general']['results_ext']) if args.benchmark else None
        
        global patterns
        for s in cfg['general']:
            if (s.startswith('extraction_regex_')):
                patterns.append(cfg['general'][s])
        if not patterns:
            fatal_error('No extraction patterns specified (introduced in mmtester 0.5.0) - check online documentation')
        
        try:
            start_time = time.time()
            for id in args.tests:
                tests_queue.put({'seed': id, 'path': inputs_path.get(id, None)})
            tests_left = args.tests
            
            def worker_loop():
                while True:
                    try:
                        seed = tests_queue.get(False)
                        result = run_test(seed)
                        results_queue.put(result)
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
            benchmark_log_scores = 0
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
                        if args.benchmark:
                            benchmark_log_scores += math.log(benchmark[seed]['score'] if benchmark[seed]['score'] > 0 else 0)
                            output += f'   Scores: {log_scores / processed : .6f} vs {benchmark_log_scores / processed : .6f}'
                        print(f'\r{output}                       ', end='', file=sys.stderr)
                        sys.stderr.flush()
                        time.sleep(0.001)
        except KeyboardInterrupt:
            print('\nInterrupted by user', file=sys.stderr)
            os._exit(1)
            
        print(file=sys.stderr)
            
        print("Time:", time.time() - start_time, file=sys.stderr)
        print("Avg Score:", sum_scores / len(results), file=sys.stderr)
        print("Avg Log Scores:", log_scores / len(results), file=sys.stderr)
    
    
if __name__ == '__main__':
    _main()