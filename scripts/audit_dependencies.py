"""Recheck official registry metadata for the reviewed lock; installs nothing."""
import json,re,urllib.request,concurrent.futures,datetime,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=ROOT/'requirements.lock'
specs=re.findall(r'^([a-z0-9-]+)==([^\s]+)',p.read_text(),re.M)
specs.extend([('torch','2.13.0+cu126'),('uv','0.12.9')])
cutoff=datetime.datetime(2026,9,15,tzinfo=datetime.timezone.utc)
def check(spec):
 name,ver=spec
 url=f'https://pypi.org/pypi/{name}/{ver.split("+")[0]}/json'
 j=json.load(urllib.request.urlopen(url))
 files=j['urls'];date=min(f['upload_time_iso_8601'] for f in files)
 assert datetime.datetime.fromisoformat(date.replace('Z','+00:00'))<cutoff,(name,date)
 if name != 'torch':
  block=re.search(r'^'+re.escape(name)+r'==.*?(?=^[a-z]|\Z)',p.read_text(),re.M|re.S)
  hashes=set(re.findall(r'--hash=sha256:([0-9a-f]{64})',block.group(0))) if block else set()
  known={f['digests']['sha256']:f for f in files}
  for digest in hashes:
   assert digest in known,(name,'unknown artifact hash',digest)
   f=known[digest]
   assert not f['yanked'],(name,'yanked artifact',f['filename'])
   assert datetime.datetime.fromisoformat(f['upload_time_iso_8601'].replace('Z','+00:00'))<cutoff,(name,'artifact too recent')
 out={'name':name,'version':ver,'metadata_url':url,'first_uploaded':date,'project_urls':j['info']['project_urls'],'advisories':j['vulnerabilities']}
 (ROOT/'work/metadata').mkdir(exist_ok=True,parents=True)
 (ROOT/f'work/metadata/{name}.json').write_text(json.dumps(j))
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex: results=list(ex.map(check,specs))
(ROOT/'dependency-audit.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'release_cutoff':'2026-09-14T00:00:00Z','packages':results},indent=2)+'\n')
for r in results:
 if r['advisories']:raise RuntimeError((r['name'],r['version'],r['advisories']))
print('Checked',len(results),'packages')
