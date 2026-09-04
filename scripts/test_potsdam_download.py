import re
import requests
import urllib3
urllib3.disable_warnings()

s = requests.Session()
s.verify = False

url = 'https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/'
r1 = s.get(url)
print('GET status:', r1.status_code)
print('GET cookies:', dict(s.cookies))

token_match = re.search(r'name="token" value="([^"]+)"', r1.text)
token = token_match.group(1) if token_match else ''
csrf_match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r1.text)
csrf_form = csrf_match.group(1) if csrf_match else s.cookies.get('sfcsrftoken', '')

print(f"Token: {token}, CSRF: {csrf_form[:10]}...")

post_data = {
    'password': 'secret',
    'csrfmiddlewaretoken': csrf_form,
    'token': token
}
headers = {'Referer': url}
r2 = s.post(url, data=post_data, headers=headers)
print('POST status:', r2.status_code)
print('POST history:', [h.status_code for h in r2.history])
print('POST final url:', r2.url)
print('POST cookies:', dict(s.cookies))
print('POST text length:', len(r2.text))

if 'Please input the password' in r2.text:
    print('Still showing password prompt.')
    err_match = re.search(r'class="error[^"]*">([^<]+)', r2.text)
    if err_match:
        print('Error msg:', err_match.group(1).strip())
else:
    print('SUCCESS! Unlocked shared folder.')
    # List files or links in the page
    links = re.findall(r'href="([^"]+)"', r2.text)
    for l in links:
        if any(ext in l.lower() for ext in ['.zip', '.7z', '.tif', 'dsm', 'top', 'potsdam', 'download', 'file']):
            print('  File link:', l)
