"""The custom-endpoint controls (Setup → Brain → Custom) stay same-origin, target fixed brain routes and hand the page
only the fields it shows: an endpoint's key never leaves the brain, only whether one is saved and its hint."""
import ast
import asyncio
import copy
import json
import unittest
from test_wipe_proxy import SOURCE, FakeHTTP, Request, Response, Session, HTTPException

KEY = 'gsk_' + 'k' * 40


def brain_status():
    """The brain's GET /api/build_f/custom answer, with fields the page must never see."""
    return {'ok': True, 'private': 'hidden',
            'endpoints': [{'id': 'groq', 'name': 'Groq', 'base_url': 'https://api.groq.com/openai/v1',
                           'json_mode': 'json_object', 'detected_at': '2026-11-18T09:00:00+00:00',
                           'models': [{'name': 'openai/gpt-oss-120b', 'context_length': 131072, 'kind': 'chat',
                                       'source': 'detected', 'raw': {'owned_by': 'OpenAI'}}],
                           'key': {'present': True, 'hint': '…kkkk', 'value': KEY}, 'reachable': True, 'error': None,
                           'headers': {'Authorization': f'Bearer {KEY}'}}],
            'in_use': {'deep': {'endpoint': 'groq', 'model': 'openai/gpt-oss-120b'}}}


class CustomProxyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.http = FakeHTTP()
        self.http.get = self.http.post
        self.ns = {'asyncio': asyncio, 'json': json, 'aiohttp': self.http, 'Request': Request,
                   'HTTPException': HTTPException, '_dialog_sessions': set(),
                   '_wipe_in_progress': False, 'HANE_URL': 'http://127.0.0.1:5201'}
        functions = {'_build_f_session_busy', '_build_f_control', '_custom_status', 'build_f_custom',
                     'build_f_custom_endpoint', 'build_f_custom_detect', 'build_f_custom_model', 'build_f_custom_use',
                     'build_f_models', 'build_f_select_models', 'build_f_preset'}
        constants = {'_CUSTOM_FIELDS', '_CUSTOM_ENDPOINT_FIELDS', '_CUSTOM_MODEL_FIELDS'}
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        nodes = [copy.deepcopy(n) for n in tree.body
                 if (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in functions)
                 or (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) in constants for t in n.targets))]
        for node in nodes:
            if hasattr(node, 'decorator_list'):
                node.decorator_list = []
        exec(compile(ast.Module(nodes, type_ignores=[]), '<custom>', 'exec'), self.ns)

    def request(self, action, value=None, **kwargs):
        return Request(body=json.dumps(value or {}).encode(),
                       headers={'x-build-f-action': action, **kwargs.pop('headers', {})}, **kwargs)

    def busy(self):
        session = Session()
        session._tts_playing = {'idx': 0}
        self.ns['_dialog_sessions'].add(session)

    def assert_page_view(self, result):
        self.assertEqual(set(result), {'ok', 'endpoints', 'in_use'})
        row = result['endpoints'][0]
        self.assertEqual(set(row), {'id', 'name', 'base_url', 'json_mode', 'detected_at', 'models', 'key',
                                    'reachable', 'error'})
        self.assertEqual(row['key'], {'present': True, 'hint': '…kkkk'})
        self.assertEqual(row['models'], [{'name': 'openai/gpt-oss-120b', 'context_length': 131072, 'kind': 'chat',
                                          'source': 'detected'}])
        self.assertEqual(result['in_use'], {'deep': {'endpoint': 'groq', 'model': 'openai/gpt-oss-120b'}})
        self.assertNotIn(KEY, json.dumps(result))

    async def test_status_reads_one_fixed_route_and_never_shows_a_key(self):
        self.http.response = Response(data=brain_status())
        self.assert_page_view(await self.ns['build_f_custom']())
        url, request = self.http.calls[0]
        self.assertEqual(url, 'http://127.0.0.1:5201/api/build_f/custom')
        self.assertFalse(request['allow_redirects'])

    async def test_status_without_a_brain_is_unavailable(self):
        for response in [Response(status=500), Response(data={'ok': False}), Response(data=['x']),
                         FakeHTTP.ClientError('refused')]:
            self.http.response = response
            with self.assertRaises(HTTPException) as caught:
                await self.ns['build_f_custom']()
            self.assertEqual(caught.exception.status_code, 503)

    async def test_endpoint_detect_and_model_go_to_their_routes_and_may_run_while_she_talks(self):
        self.busy()
        value = {'id': 'groq', 'name': 'Groq', 'base_url': 'https://api.groq.com/openai/v1', 'key': KEY}
        for fn, route, action in [('build_f_custom_endpoint', '/api/build_f/custom/endpoint', 'custom-endpoint'),
                                  ('build_f_custom_detect', '/api/build_f/custom/detect', 'custom-detect'),
                                  ('build_f_custom_model', '/api/build_f/custom/model', 'custom-model')]:
            self.http.calls.clear()
            self.http.response = Response(data=brain_status())
            self.assert_page_view(await self.ns[fn](self.request(action, value)))
            url, request = self.http.calls[0]
            self.assertEqual(url, 'http://127.0.0.1:5201' + route)
            self.assertEqual(request['headers'], {'X-Build-F-Action': action})
            self.assertEqual(request['json'], value)
            self.assertFalse(request['allow_redirects'])
            with self.assertRaises(HTTPException) as caught:
                await self.ns[fn](self.request('custom-use', value))
            self.assertEqual(caught.exception.status_code, 403)
            with self.assertRaises(HTTPException) as caught:
                await self.ns[fn](self.request(action, value, origin='https://evil.test'))
            self.assertEqual(caught.exception.status_code, 403)
        self.assertFalse(self.ns['_wipe_in_progress'])

    async def test_use_waits_for_her_and_passes_the_brains_busy_refusal_on(self):
        value = {'roles': {'deep': {'endpoint': 'groq', 'model': 'openai/gpt-oss-120b'}}}
        self.http.response = Response(data={'ok': True, 'selections': {'deep': 'deep:custom:groq:openai/gpt-oss-120b'},
                                            'transport': 'mixed', 'in_use': {}, 'options': {'private': 1}})
        result = await self.ns['build_f_custom_use'](self.request('custom-use', value))
        self.assertEqual(set(result), {'ok', 'selections', 'transport', 'in_use'})
        url, request = self.http.calls[0]
        self.assertEqual(url, 'http://127.0.0.1:5201/api/build_f/custom/use')
        self.assertEqual(request['headers'], {'X-Build-F-Action': 'custom-use'})
        self.http.response = Response(status=409, data={'ok': False, 'code': 'model_busy', 'error': 'Model selection is busy'})
        with self.assertRaises(HTTPException) as caught:
            await self.ns['build_f_custom_use'](self.request('custom-use', value))
        self.assertEqual((caught.exception.status_code, caught.exception.detail),
                         (409, {'code': 'model_busy', 'message': 'Model selection is busy'}))
        self.busy()
        self.http.calls.clear()
        with self.assertRaises(HTTPException) as caught:
            await self.ns['build_f_custom_use'](self.request('custom-use', value))
        self.assertEqual(caught.exception.status_code, 409)
        self.assertFalse(self.http.calls)

    async def test_the_model_choices_carry_the_custom_endpoints(self):
        custom = {'endpoints': [{'id': 'groq', 'name': 'Groq', 'base_url': 'https://api.groq.com/openai/v1',
                                 'json_mode': 'json_object', 'models': ['openai/gpt-oss-120b']}]}
        self.http.response = Response(data={'ok': True, 'selections': {}, 'options': {}, 'ollama': {}, 'custom': custom})
        self.assertEqual((await self.ns['build_f_models']())['custom'], custom)
        for fn, action in [('build_f_select_models', 'select-models'), ('build_f_preset', 'apply-preset')]:
            self.assertEqual((await self.ns[fn](self.request(action, {'selections': {}})))['custom'], custom)


if __name__ == '__main__':
    unittest.main()
