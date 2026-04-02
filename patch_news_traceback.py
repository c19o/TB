"""Patch feature_library.py to add traceback to news feature error handler."""
import sys

path = '/workspace/v3.3/feature_library.py'
with open(path, 'r') as f:
    lines = f.readlines()

# Find the except block near "News feature computation"
found = False
for i, line in enumerate(lines):
    if 'News feature computation failed' in line:
        # Find the except line above it
        exc_line = i
        while exc_line > 0 and 'except Exception' not in lines[exc_line]:
            exc_line -= 1

        # Remove everything from exc_line+1 to (and including) the line with 'News feature'
        # Also remove any continuation lines
        end = i + 1
        while end < len(lines) and lines[end].strip().startswith(('+', "'", 'Traceback')):
            end += 1

        new_lines = [
            '            import logging as _lg, traceback as _tb\n',
            '            _msg = "News feature computation failed: " + repr(_e)\n',
            '            _msg += "\\nTraceback:\\n" + _tb.format_exc()\n',
            '            _lg.getLogger("feature_library").warning(_msg)\n',
        ]

        lines[exc_line+1:end] = new_lines
        found = True
        print(f'Patched: replaced lines {exc_line+2}-{end} with {len(new_lines)} lines')
        break

if not found:
    print('ERROR: Pattern not found')
    sys.exit(1)

with open(path, 'w') as f:
    f.writelines(lines)
print('Saved successfully')
