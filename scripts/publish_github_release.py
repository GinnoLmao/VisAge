"""Publish this user's authorized VisAge source/release using existing Git auth.

Credentials are read in memory from Git Credential Manager and never printed,
written to disk or included in request URLs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = 'GinnoLmao/VisAge'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect', 'create', 'release', 'upload', 'pages'])
    parser.add_argument('--file', type=Path)
    args = parser.parse_args()
    environment = os.environ.copy()
    environment.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\nusername=GinnoLmao\n\n',
        capture_output=True, text=True, check=True, env=environment)
    credentials = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    headers = {'Authorization': 'Bearer ' + credentials['password'], 'Accept': 'application/vnd.github+json',
        'User-Agent': 'VisAge-delivery', 'X-GitHub-Api-Version': '2022-11-28'}
    def request(path, method='GET', value=None):
        body = json.dumps(value).encode() if value is not None else None
        req = urllib.request.Request('https://api.github.com' + path, data=body, method=method,
            headers={**headers, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response) if response.status != 204 else {}
        except urllib.error.HTTPError as exc:
            if exc.code == 404: return None
            raise RuntimeError(f'GitHub operation failed with HTTP {exc.code}: ' + exc.read().decode()[:500]) from None
    user = request('/user')
    if user['login'] != 'GinnoLmao': raise ValueError('Unexpected authenticated GitHub account.')
    repo = request('/repos/' + REPOSITORY)
    if args.action == 'create' and repo is None:
        repo = request('/user/repos', 'POST', {'name':'VisAge', 'description':'Three pretrained vision models for age-group estimation, with a FastAPI web app and reproducible evaluation.',
            'private':False, 'auto_init':False, 'has_issues':True})
    if repo is None: raise ValueError('Create the authorized repository first.')
    if repo['private']: raise ValueError('The requested repository must be public.')
    output = {'repository':repo['html_url'], 'public':not repo['private'], 'id':repo['id']}
    if args.action in ('release', 'upload'):
        release = request('/repos/' + REPOSITORY + '/releases/tags/v0.4.0')
        if args.action == 'release' and release is None:
            release = request('/repos/' + REPOSITORY + '/releases', 'POST', {'tag_name':'v0.4.0','target_commitish':'main',
                'name':'VisAge three-model web application v0.4.0', 'draft':False, 'prerelease':False,
                'body':'Three selected seed-42 classifiers from six completed runs. Includes checksum-bound model/detector runtime assets and fresh local delivery. No raw face photos or optimizer states. ConvNeXt weights retain CC BY-NC 4.0 restrictions; see THIRD_PARTY_NOTICES.md. Pilot testing and further tuning were excluded by the user.'})
        if release is None: raise ValueError('Create the release before uploading assets.')
        output['release'] = release['html_url']
        if args.action == 'upload':
            if not args.file or not args.file.is_file() or args.file.suffix not in ('.zip','.pdf','.json'):
                raise ValueError('A reviewed release file is required.')
            existing = next((a for a in release['assets'] if a['name'] == args.file.name), None)
            if existing: raise FileExistsError('A release asset with this name already exists; do not overwrite it.')
            import http.client
            path = f'/repos/{REPOSITORY}/releases/{release["id"]}/assets?name=' + urllib.parse.quote(args.file.name)
            connection = http.client.HTTPSConnection('uploads.github.com', timeout=300)
            connection.putrequest('POST', path)
            for key, value in {**headers, 'Content-Type':'application/octet-stream', 'Content-Length':str(args.file.stat().st_size)}.items(): connection.putheader(key,value)
            connection.endheaders()
            with args.file.open('rb') as source:
                while block := source.read(1024 * 1024): connection.send(block)
            response = connection.getresponse(); body = response.read(); connection.close()
            if response.status != 201: raise RuntimeError('Release upload failed with HTTP ' + str(response.status))
            asset = json.loads(body); output['asset_url'] = asset['browser_download_url']; output['bytes'] = asset['size']
    if args.action == 'pages':
        pages = request('/repos/' + REPOSITORY + '/pages')
        if pages is None: pages = request('/repos/' + REPOSITORY + '/pages', 'POST', {'build_type':'legacy','source':{'branch':'main','path':'/docs'}})
        output['pages_url'] = pages['html_url']; output['pages_status'] = pages['status']
    print(json.dumps(output), flush=True)


if __name__ == '__main__': main()
