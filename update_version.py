import os
import re
import sys
import shutil

assert len(sys.argv) == 2
version = sys.argv[1]

assert re.match('^[0-9]+\.[0-9]+\.[0-9]+', version)

with open('mmtester/__init__.py', 'w') as f:
    f.write(f"__version__ = '{version}'\n")
    
with open('mmtester/tester.cfg', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith('version'):
        lines[i] = f'version = {version}\n'
        break

with open('mmtester/tester.cfg', 'w') as f:
    f.writelines(lines)
    
shutil.copy('mmtester/tester.cfg', 'mmtester/backup.cfg')
