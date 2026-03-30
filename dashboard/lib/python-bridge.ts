import { execSync } from 'child_process';
import path from 'path';

const PROJECT_DIR = path.join(process.cwd(), '..');

/**
 * Call the Python api_bridge.py and return parsed JSON.
 */
export function callPython(command: string, ...args: string[]): any {
  const allArgs = [command, ...args].map(a => `"${a}"`).join(' ');
  const cmd = `python api_bridge.py ${allArgs}`;

  try {
    const stdout = execSync(cmd, {
      cwd: PROJECT_DIR,
      encoding: 'utf-8',
      timeout: 60000,
      windowsHide: true,
    });

    // The Python script outputs JSON on the last line of stdout.
    // Filter out any stray warnings/prints by finding the last JSON line.
    const lines = stdout.trim().split('\n');
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line.startsWith('{') || line.startsWith('[')) {
        return JSON.parse(line);
      }
    }

    throw new Error('No JSON found in Python output');
  } catch (err: any) {
    // If execSync threw, check if there's stderr info
    const message = err.stderr
      ? `Python error: ${err.stderr.toString().slice(0, 500)}`
      : `Python bridge error: ${err.message}`;
    throw new Error(message);
  }
}
