from common import *
import subprocess
if __name__=='__main__':
 assert sys.argv[1] in ['infer','analyze_verify']
 stage=sys.argv[1];attempt=len(list((WORK/'logs').glob(stage+'_*.stdout.txt')))+1;cmd=[str(ROOT/'.venv/bin/python'),'-B',str(WORK/'code'/f'{stage}.py')]
 log('stage_start',stage=stage,attempt=attempt,subprocess_command=cmd,code_sha256=sha(WORK/'code'/f'{stage}.py'))
 with (WORK/f'logs/{stage}_{attempt:03d}.stdout.txt').open('x') as out,(WORK/f'logs/{stage}_{attempt:03d}.stderr.txt').open('x') as err:p=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=err,env=os.environ)
 log('stage_end',stage=stage,attempt=attempt,returncode=p.returncode)
 print(dumps(read('STATUS.json')));sys.exit(p.returncode)
