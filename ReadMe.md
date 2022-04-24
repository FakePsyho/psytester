Simple cmd-line tester for Topcoder Marathon & AtCoder Heuristic Contests. 

Install the latest version with `pip install mmtester --upgrade`

## Main Features
- Works with both [Topcoder](https://www.topcoder.com/home) Marathons and [AtCoder](https://atcoder.jp/) Heuristic Contests
- Run batches of tests using multiple threads and store results in a friendly JSON format
- Print the local scoreboard
- Group/Filter results by user-defined metadata
- Search through tests according to some metrics (useful for finding tests close to timing out or some more extreme test cases)

## Quick Setup: Topcoder Marathons
- Download official jar tester
- Run `mmtester --config-load topcoder` in problem's directory (note: if you're not using the [new](https://docs.google.com/document/d/100JX1dgENRlrxt3tZfVWQ70PxoOQjyl1LfxlvnWxRG8) tester use `mmtester --config-load old_topcoder` instead)
- Edit newly created tester.cfg config file to change:
	- number of threads (`default/threads_no`)
	- `-exec` part to match the name of your executable (`general/cmd_tester`)
	- change the type of scoring function used ('default/scoring'); by default the scoring function used is max (YOUR_SCORE / BEST_SCORE), you can use min (BEST_SCORE / YOUR_SCORE) or raw (YOUR_SCORE); 


## Quick Setup: AtCoder Heuristic Contests
- Download official tools
- Run `mmtester --config-load atcoder` in problem's directory
- Edit newly created tester.cfg config file to change the number of threads (`default/threads_no`) and the command that invokes provided tester and output the scores (unfortunately AHCs do not have standarized tools across different contests)
- Few examples:
	- [AHC008](https://atcoder.jp/contests/ahc008): `cmd_tester = cargo run --bin tester a.exe < in/%SEED04%.txt > %OUTPUT_DIR%/%SEED%.out 2> %OUTPUT_DIR%/%SEED%.err`
	- [AHC009](https://atcoder.jp/contests/ahc009): `cmd_tester = a.exe < in/%SEED04%.txt > %OUTPUT_DIR%/%SEED%.tmp 2> %OUTPUT_DIR%/%SEED%.err && vis.exe in/%SEED04%.txt %OUTPUT_DIR%/%SEED%.tmp > %OUTPUT_DIR%/%SEED%.out`
- Note that `output_files` should be a list of files that mmtester scans in order to extract score (in form of `Score = value`) and/or user-defined metadata (in form of `[DATA] variable = value`); if you might to modify `output_files` as well (given examples do not require that).
- mmtester doesn't currently use the generator so you have to generate the tests manualy (the generator wrapper is implemented, but it had some undocumented side-effects so I decided to turn it temporarily)

## Basic Workflow
- Test your solution manually (without mmtester) and verify that it's working correctly.
- Verify that mmtester is set up correctly by executing a small test run: `mmtester -t 10`. This should print results to standard output
- Run `mmtester [results_file]` in order to run a batch of tests and create a `results_file.res` with scores. Output of each test is redirected to a text file and then later the file is scanned in order to extract the score (and potentially other variables). Results file will contain scores for each test case in JSON format. By default, output files will be stored at `./tests/results_file/`.
- Run `mmtester -s` to see the scoreboard for all of the results' files in the current directory. 
- By default, results files contains only Score and the test ID (seed). You can add additional metadata by printing `[DATA] X = Y` (spaces are mandatory) in your solution. For example, let's say the the problem is on the square grid of size N. If your solution prints `[DATA] N = ...`, you can run `mmtester -s --data [results_file_with_data] --groups N` to see the summary grouped by different sizes of the grid. You can also manually specify multiple groups with different criteria like `--groups N=2 N=3-5 N=6-10 D=2 D=4`. Similarly, you can use `--filters N=5-10` to filter the testcases to include only those meeting specific criteria.
- For more options/features, please look at the config comments and/or built-in help. You can override most of the options by specifying adequate command line option.

## Mode: Run Tests
- *Information explaining how it works incoming*
- Use `--tests A-B` to select the range of seeds to use; `--tests N` is equivalent to `--tests 1-N`;
- Use `--m threads_no` to change the number of threads, you can also set it in the config file (`default/threads_no`)
- *More info coming*
 
## Mode: Show
- Running `mmtester --show` will print results table based results files in current directory
- By default results are sorted by file name, you can also sort based on time by changing config option `default/sorting` to `date`
- There are config options to enable/disable showing errors/bests/uniques
- If you want to remove some old results file the easiest way is to just move/delete them
- You can use `--groups ...` to add additional columns that will show the score per group; Similarly you can use `--filters ...`. Both of those options require adding metadata by adding `[DATA] variable = value` to your output files. For more info check [grouping and filtering](https://github.com/FakePsyho/mmtester/#grouping-and-filtering)
- *More info coming*

## Mode: Find
- Running `mmtester --find results_file var(+/-) [limit]` sorts the tests by `var`(ascending / descending) and prints them (either all or up to `limit`). You can use `LATEST` as a `results_file` in order to use the latest results file. For example, you can run `mmtester --find LATEST time- 10`to print 10 tests from the latest run that had the longest execution time.
- You can combine it with `--filters ...` options
- When running tests, you can use `--tests test_file` to run tests on a specific subset of seeds. This allows you to easily rerun a solution on a specific subset of tests: `mmtester --find ... > test_subset && mmtester --tests test_subset ...`

## Mode: Generate Scripts
Script generation is experimental and was broken so many times that it's been temporarily disabled. 

## Grouping and Filtering
- *I know it's the "killer feature" of the mmtester, but more info is incoming, I promise!*

## Config Files
- *More info coming*

## Tips
- Do not expect the same execution speed when running concurently with multiple threads. 
- *More info coming*

## Known Issues
- Currently mmtester is not backward compatible with old config files, you'll have to manually update the config file to a new version if you update mmtester while working on the same problem. This should change in the future versions.
- There's a tiny glitch where on some machines sometimes progress is double printed on the same line.
- *More issues coming ;)*