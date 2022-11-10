
import sys
import pyrootutils

root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)
sys.path.insert(0,'serve/ts_scripts/')
import unittest

import requests
import json
import base64
from requests import Response
from serve.ts_scripts.torchserve_grpc_client import infer, get_inference_stub

class TestFargateGradio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:8080/predictions/cifar"


        cls.image_paths = ['1000_truck.png',  '10011_cat.png',  '10010_airplane.png',  '10008_airplane.png',  '10001_frog.png',  '10003_ship.png',  '10009_frog.png',  '10006_deer.png',  '10005_cat.png',  '10007_frog.png']
        
        cls.stub = get_inference_stub()
        # convert image to base64

    def test_predict(self):
        for image_path in self.image_paths:
            print(f"testing: {image_path}")

            # data = open('tests/image/' + image_path, 'rb').read()
            response = infer(self.stub, 'cifar', 'tests/image/' + image_path)
                # ext = image_path.split('.')[-1]

                # prefix = f'data:image/{ext};base64,'
                # base64_data = prefix + base64.b64encode(f.read()).decode('utf-8')

            # payload = json.dumps({
            # "data": [
            #     base64_data
            # ]
            # })

            # headers = {
            # 'Content-Type': 'application/json'
            # }

            # response: Response = requests.request("POST", self.base_url, data=data, timeout=15)

            # print(f"response: {response.text}")

            data = json.loads(response)

            predicted_label = list(data)[0]
            act_label = image_path.split(".")[0].split('_')[-1]

            print(f"predicted label: {predicted_label}, actual label: {act_label}")

            self.assertEqual(act_label, predicted_label)

            print(f"done testing: {image_path}")

            print()


if __name__ == '__main__':
    unittest.main()