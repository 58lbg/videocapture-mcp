from fastmcp import Context, FastMCP

mcp = FastMCP("my-mc")

@mcp.tool()
def read_text_file(file_path: str) -> str:
    """
    读取文本文件的全部内容并返回字符串。

    :param file_path: 文件路径
    :return: 文件内容字符串
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"❌ 文件未找到: {file_path}")
        return ""
    except Exception as e:
        print(f"⚠️ 读取文件时出错: {e}")
        return ""



mcp.run(transport="streamable-http", host="10.253.69.100", port=9002)