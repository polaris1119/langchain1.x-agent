"""
和风天气 API 服务
负责与和风天气 API 进行交互

API 文档:
- 城市查询: https://dev.qweather.com/docs/api/geoapi/city-lookup/
- 实时天气: https://dev.qweather.com/docs/api/weather/weather-now/
"""
import httpx
from typing import Optional, Dict, Any
from weather_agent.core.config import settings


class QWeatherService:
    """和风天气服务类"""

    def __init__(self):
        self.api_key = settings.QWEATHER_API_KEY
        self.base_url = settings.QWEATHER_BASE_URL
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_city_id(self, city_name: str) -> Optional[str]:
        """
        通过城市名称获取城市 ID

        接口文档: https://dev.qweather.com/docs/api/geoapi/city-lookup/

        Args:
            city_name: 城市名称，例如"北京"、"上海"

        Returns:
            城市ID，如果未找到返回 None
        """
        # 使用正确的 geo API 路径
        url = f"{self.base_url}/geo/v2/city/lookup"
        params = {
            "location": city_name,
            "key": self.api_key
        }

        try:
            response = await self.client.get(url, params=params)
            data = response.json()

            if data.get("code") == "200" and data.get("location"):
                return data["location"][0]["id"]
            return None
        except Exception as e:
            print(f"获取城市ID失败: {e}")
            return None

    async def get_current_weather(self, city_name: str) -> Dict[str, Any]:
        """
        获取指定城市的实时天气

        接口文档: https://dev.qweather.com/docs/api/weather/weather-now/

        Args:
            city_name: 城市名称

        Returns:
            天气数据字典
        """
        # 1. 获取城市 ID
        city_id = await self.get_city_id(city_name)
        if not city_id:
            return {"error": f"未找到城市: {city_name}"}

        # 2. 使用正确的天气 API 路径
        url = f"{self.base_url}/v7/weather/now"
        params = {
            "location": city_id,
            "key": self.api_key
        }

        try:
            response = await self.client.get(url, params=params)
            data = response.json()

            if data.get("code") == "200":
                return {
                    "city": city_name,
                    "temp": data["now"]["temp"],
                    "feels_like": data["now"]["feelsLike"],
                    "weather": data["now"]["text"],
                    "wind_dir": data["now"]["windDir"],
                    "wind_scale": data["now"]["windScale"],
                    "humidity": data["now"]["humidity"],
                    "precip": data["now"]["precip"],
                }
            else:
                return {"error": f"获取天气失败: {data.get('code')}"}
        except Exception as e:
            return {"error": f"请求失败: {str(e)}"}

    async def get_forecast(self, city_name: str, days: int = 3) -> Dict[str, Any]:
        """
        获取指定城市的天气预报

        Args:
            city_name: 城市名称
            days: 预报天数 (1-7)

        Returns:
            天气预报数据字典
        """
        city_id = await self.get_city_id(city_name)
        if not city_id:
            return {"error": f"未找到城市: {city_name}"}

        # 和风天气免费版支持 3 天预报
        url = f"{self.base_url}/v7/weather/{min(days, 3)}d"
        params = {
            "location": city_id,
            "key": self.api_key
        }

        try:
            response = await self.client.get(url, params=params)
            data = response.json()

            if data.get("code") == "200":
                forecasts = []
                for day in data.get("daily", []):
                    forecasts.append({
                        "date": day["fxDate"],
                        "temp_max": day["tempMax"],
                        "temp_min": day["tempMin"],
                        "weather_day": day["textDay"],
                        "weather_night": day["textNight"],
                        "wind_dir_day": day["windDirDay"],
                        "wind_scale_day": day["windScaleDay"],
                    })

                return {
                    "city": city_name,
                    "forecasts": forecasts
                }
            else:
                return {"error": f"获取预报失败: {data.get('code')}"}
        except Exception as e:
            return {"error": f"请求失败: {str(e)}"}

    async def format_weather_text(self, city_name: str, days: int = 1) -> str:
        """
        格式化天气信息为自然语言文本

        Args:
            city_name: 城市名称
            days: 查询天数

        Returns:
            格式化的天气描述文本
        """
        # 获取实时天气
        current = await self.get_current_weather(city_name)
        if "error" in current:
            return current["error"]

        result = [
            f"📍 {current['city']}当前天气：",
            f"🌡️ 温度: {current['temp']}°C (体感 {current['feels_like']}°C)",
            f"☁️ 天气: {current['weather']}",
            f"💨 风向: {current['wind_dir']} {current['wind_scale']}级",
            f"💧 湿度: {current['humidity']}%",
        ]

        # 如果需要多天预报
        if days > 1:
            forecast = await self.get_forecast(city_name, days - 1)
            if "forecasts" in forecast:
                result.append("\n📅 未来天气预报：")
                for day in forecast["forecasts"]:
                    result.append(
                        f"  {day['date']}: {day['temp_min']}-{day['temp_max']}°C, "
                        f"{day['weather_day']}, {day['wind_dir_day']} {day['wind_scale_day']}级"
                    )

        return "\n".join(result)

    async def close(self):
        """关闭 HTTP 客户端"""
        await self.client.aclose()
