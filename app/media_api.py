import aiohttp
import json
from typing import Dict, Any


class MediaMtxAPI:
    def __init__(self, base_url: str = "http://mediamtx:9999"):
        self.base_url = base_url.rstrip('/')

    async def create_path(self, path_name: str) -> bool:
        """Создает новый путь в медиамукс через API"""
        url = f"{self.base_url}/v1/paths/add"
        data = {
            "name": path_name,
            "source": "publisher"
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=data) as response:
                    if response.status in (200, 201):
                        print(f"Путь создан: {path_name}")
                        return True
                    else:
                        text = await response.text()
                        print(f"Ошибка создания пути {path_name}: {text}")
                        return False
            except Exception as e:
                print(f"Ошибка подключения к медиамукс API: {e}")
                return False

    async def delete_path(self, path_name: str) -> bool:
        """Удаляет путь из медиамукс"""
        url = f"{self.base_url}/v1/paths/delete/{path_name}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url) as response:
                    if response.status in (200, 204):
                        print(f"Путь удален: {path_name}")
                        return True
                    else:
                        text = await response.text()
                        print(f"Ошибка удаления пути {path_name}: {text}")
                        return False
            except Exception as e:
                print(f"Ошибка подключения к медиамукс API: {e}")
                return False

    async def list_paths(self) -> Dict[str, Any]:
        """Получает список всех путей"""
        url = f"{self.base_url}/v1/paths/list"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {}
            except Exception as e:
                print(f"Ошибка получения списка путей: {e}")
                return {}