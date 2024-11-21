from locust import HttpUser, TaskSet, task, between
import uuid

class UserBehavior(TaskSet):
    @task
    def interact(self):
        random_uuid = str(uuid.uuid4())
        with self.client.post("/interact", json={
            'uuid': random_uuid,
            'scene': 'qwen-14b',
            'system': '请判断用户意图，如果用户想听歌，请输出：听歌(<歌名>)。例如用户输入为“帮我放一首海阔天空”，你的输出应该是“听歌(海阔天空)”。如果用户不想听歌，例如输入为“帮我查下天气”，你的输出应该是“无”。',
            'user': '我想听左手指月那首歌'
        }, catch_response=True) as response:
            print(response)


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(0,1)  # 每个用户请求之间等待1到3秒
    host = "http://localhost:8594"  # 设置服务的IP和端口


