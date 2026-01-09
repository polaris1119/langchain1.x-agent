"""
终端测试界面
提供交互式命令行界面用于测试天气查询 Agent
"""
import asyncio
from weather_agent.agent.weather_agent import WeatherAgent


async def main():
    """主函数"""
    agent = WeatherAgent()

    print("=" * 50)
    print("🌤️  天气查询 Agent - 终端测试界面")
    print("=" * 50)
    print("\n欢迎使用天气查询助手！")
    print("你可以问我任何城市的天气情况。")
    print("\n示例:")
    print("  - 北京今天天气怎么样?")
    print("  - 上海未来三天天气")
    print("  - 深圳现在多少度?")
    print("\n输入 'quit' 或 'exit' 退出程序")
    print("-" * 50)

    while True:
        try:
            # 获取用户输入
            user_input = "\n您: ".strip()

            # 使用 input() 获取用户输入
            import sys
            if sys.stdin.isatty():
                user_input = input("您: ").strip()
            else:
                # 非交互模式
                break

            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n👋 再见！感谢使用天气查询助手！")
                break

            # 跳过空输入
            if not user_input:
                continue

            # 调用 Agent
            print("\nAgent: ", end="", flush=True)
            response = await agent.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\n👋 程序已中断，再见！")
            break
        except EOFError:
            print("\n\n👋 输入结束，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
