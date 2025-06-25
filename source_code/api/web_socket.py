# import asyncio
# import threading
# import uuid
# from typing import Dict

# import websockets

# server = None


# class WebSocketServer:
#     def __init__(self, host: str = "0.0.0.0", port: int = 8765):
#         self.host = host
#         self.port = port
#         self.expert_chat_connected_clients: Dict[str] = {}
#         self.rag_progress_connected_clients: Dict[str] = {}
#         self.server = None

#     async def register_client(self, websocket):
#         """
#         注册新客户端连接
#         客户端ID格式为：
#         ws://<ip:port>/yixiaozhu_api_websocket/<client_type>/<client_id>
#         """
#         try:
#             client_type, client_id = websocket.request.path.split("/")[-2], websocket.request.path.split("/")[-1]
#             if client_type not in ["expert_chat", "rag_progress"]:
#                 raise ValueError(f"Invalid client_type: {client_type}")
#         except Exception as e:
#             print(f"Invalid client_id: {client_id} or client_type: {client_type}")
#             raise e
#         if client_type == "expert_chat":
#             self.expert_chat_connected_clients[client_id] = websocket
#             return client_id
#         elif client_type == "rag_progress":
#             self.rag_progress_connected_clients[client_id] = websocket
#             return client_id
#         else:
#             raise ValueError(f"Invalid client_type: {client_type}")

#     async def unregister_client(self, client_id: str):
#         """移除断开连接的客户端"""
#         if client_id in self.expert_chat_connected_clients:
#             del self.expert_chat_connected_clients[client_id]
#             print(f"Client disconnected: {client_id}. Total clients: {len(self.expert_chat_connected_clients)}")
#         elif client_id in self.rag_progress_connected_clients:
#             del self.rag_progress_connected_clients[client_id]
#             print(f"Client disconnected: {client_id}. Total clients: {len(self.rag_progress_connected_clients)}")
#         else:
#             print(f"Client not found: {client_id}")

#     async def handler(self, websocket):
#         """处理客户端连接"""
#         client_id = await self.register_client(websocket)
#         try:
#             async for message in websocket:
#                 # 这里可以处理客户端发来的消息
#                 print(f"Received message from {client_id}: {message}")
#                 # 可以回复消息
#                 # await websocket.send(f"Server received: {message}")
#         except websockets.exceptions.ConnectionClosed:
#             pass
#         finally:
#             await self.unregister_client(client_id)

#     async def broadcast(self, message: str):
#         """广播消息给所有已连接的专家聊天类型websocket客户端"""
#         if not self.expert_chat_connected_clients:
#             print("No clients connected to broadcast to.")
#             return

#         disexpert_chat_connected_clients = []
#         coros = []
#         for client_id, websocket in self.expert_chat_connected_clients.items():
#             coros.append(self._safe_send(websocket, client_id, message, disexpert_chat_connected_clients))
#         await asyncio.gather(*coros)

#         for client_id in disexpert_chat_connected_clients:
#             await self.unregister_client(client_id)

#     async def _safe_send(self, websocket, client_id, message, disexpert_chat_connected_clients):
#         try:
#             await websocket.send(message)
#             print(f"Message sent to {client_id}")
#         except websockets.exceptions.ConnectionClosed:
#             disexpert_chat_connected_clients.append(client_id)

#     async def send_message(self, client_id: str, message: str):
#         """向指定客户端发送消息"""
#         if client_id not in self.expert_chat_connected_clients  and client_id not in self.rag_progress_connected_clients:
#             print(f"Client {client_id} not found.")
#             return
#         try:
#             if client_id in self.expert_chat_connected_clients:
#                 await self.expert_chat_connected_clients[client_id].send(message)
#                 print(f"Message sent to {client_id}")
#             elif client_id in self.rag_progress_connected_clients:
#                 await self.rag_progress_connected_clients[client_id].send(message)
#             else:
#                 print(f"Client {client_id} disconnected")
#         except websockets.exceptions.ConnectionClosed:
#             print(f"Client {client_id} disconnected.")
#             await self.unregister_client(client_id)

#     async def start(self):
#         """启动WebSocket服务器"""
#         self.server = await websockets.serve(self.handler, self.host, self.port)
#         print(f"WebSocket server started on ws://{self.host}:{self.port}")

#     async def stop(self):
#         if self.server:
#             self.server.close()
#             await self.server.wait_closed()
#             print("WebSocket server stopped")
#         self.expert_chat_connected_clients.clear()


# # 使用示例
# async def main():
#     global server
#     try:
#         server = WebSocketServer()
#         await server.start()
#         while True:
#             await asyncio.sleep(30)
#             print("websocket server is running")
#     except KeyboardInterrupt:
#         await server.stop()


# def start_async_in_thread():
#     asyncio.run(main())


# thread = threading.Thread(target=start_async_in_thread, name="AsyncThread")
# thread.start()

import asyncio
import logging
import signal
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('websocket_server')

# 全局服务器实例
server = None


@dataclass
class ClientInfo:
    """客户端信息数据类"""
    id: str
    type: str
    websocket: WebSocketServerProtocol
    connected_at: datetime = field(default_factory=datetime.now)


class ClientManager:
    """客户端连接管理器"""
    def __init__(self):
        self.expert_chat_clients: Dict[str, WebSocketServerProtocol] = {}
        self.rag_progress_clients: Dict[str, WebSocketServerProtocol] = {}
        self.valid_client_types = ["expert_chat", "rag_progress"]
    
    def add_client(self, client_type: str, client_id: str, websocket: WebSocketServerProtocol) -> None:
        """添加客户端连接"""
        if client_type not in self.valid_client_types:
            raise ValueError(f"Invalid client_type: {client_type}")
            
        if client_type == "expert_chat":
            self.expert_chat_clients[client_id] = websocket
            logger.info(f"Expert chat client {client_id} connected. Total: {len(self.expert_chat_clients)}")
        elif client_type == "rag_progress":
            self.rag_progress_clients[client_id] = websocket
            logger.info(f"RAG progress client {client_id} connected. Total: {len(self.rag_progress_clients)}")
    
    def remove_client(self, client_id: str) -> bool:
        """移除客户端连接，返回是否成功移除"""
        if client_id in self.expert_chat_clients:
            del self.expert_chat_clients[client_id]
            logger.info(f"Expert chat client {client_id} disconnected. Total: {len(self.expert_chat_clients)}")
            return True
        elif client_id in self.rag_progress_clients:
            del self.rag_progress_clients[client_id]
            logger.info(f"RAG progress client {client_id} disconnected. Total: {len(self.rag_progress_clients)}")
            return True
        else:
            logger.warning(f"Client {client_id} not found for removal")
            return False
    
    def get_client(self, client_id: str) -> Optional[WebSocketServerProtocol]:
        """获取客户端连接"""
        if client_id in self.expert_chat_clients:
            return self.expert_chat_clients[client_id]
        elif client_id in self.rag_progress_clients:
            return self.rag_progress_clients[client_id]
        return None
    
    def get_client_type(self, client_id: str) -> Optional[str]:
        """获取客户端类型"""
        if client_id in self.expert_chat_clients:
            return "expert_chat"
        elif client_id in self.rag_progress_clients:
            return "rag_progress"
        return None
    
    def get_clients_by_type(self, client_type: str) -> Dict[str, WebSocketServerProtocol]:
        """获取指定类型的所有客户端"""
        if client_type not in self.valid_client_types:
            raise ValueError(f"Invalid client_type: {client_type}")
            
        if client_type == "expert_chat":
            return self.expert_chat_clients
        elif client_type == "rag_progress":
            return self.rag_progress_clients
        return {}
    
    def clear_all(self) -> None:
        """清空所有客户端连接"""
        self.expert_chat_clients.clear()
        self.rag_progress_clients.clear()
        logger.info("All client connections cleared")


class WebSocketServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.client_manager = ClientManager()
        self.server = None

    async def register_client(self, websocket: WebSocketServerProtocol) -> str:
        """
        注册新客户端连接
        客户端ID格式为：
        ws://<ip:port>/yixiaozhu_api_websocket/<client_type>/<client_id>
        """
        try:
            path_parts = websocket.request.path.split("/")
            if len(path_parts) < 2:
                raise ValueError(f"Invalid path format: {websocket.request.path}")
            
            client_type, client_id = path_parts[-2], path_parts[-1]
            if client_type not in self.client_manager.valid_client_types:
                raise ValueError(f"Invalid client_type: {client_type}")
            
            self.client_manager.add_client(client_type, client_id, websocket)
            return client_id
        except ValueError as e:
            logger.error(f"Registration error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during client registration: {str(e)}")
            raise

    async def unregister_client(self, client_id: str) -> None:
        """移除断开连接的客户端"""
        self.client_manager.remove_client(client_id)

    async def handler(self, websocket: WebSocketServerProtocol) -> None:
        """处理客户端连接"""
        client_id = None
        try:
            client_id = await self.register_client(websocket)
            async for message in websocket:
                # 这里可以处理客户端发来的消息
                logger.info(f"Received message from {client_id}: {message}")
                # 可以回复消息
                # await websocket.send(f"Server received: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for client {client_id}")
        except Exception as e:
            logger.error(f"Error handling client connection: {str(e)}")
        finally:
            if client_id:
                await self.unregister_client(client_id)

    async def broadcast(self, message: str, client_type: str = "expert_chat") -> None:
        """广播消息给指定类型的所有已连接客户端"""
        try:
            clients = self.client_manager.get_clients_by_type(client_type)
            
            if not clients:
                logger.info(f"No {client_type} clients connected to broadcast to.")
                return

            disconnected_clients: List[str] = []
            coros = []
            for client_id, websocket in clients.items():
                coros.append(self._safe_send(websocket, client_id, message, disconnected_clients))
            
            await asyncio.gather(*coros)

            for client_id in disconnected_clients:
                await self.unregister_client(client_id)
        except Exception as e:
            logger.error(f"Error broadcasting message: {str(e)}")

    async def _safe_send(self, websocket: WebSocketServerProtocol, client_id: str, 
                         message: str, disconnected_clients: List[str]) -> None:
        """安全地发送消息，处理连接关闭的情况"""
        try:
            await websocket.send(message)
            logger.info(f"Message sent to {client_id}")
        except websockets.exceptions.ConnectionClosed:
            disconnected_clients.append(client_id)
            logger.info(f"Connection closed while sending to {client_id}")
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {str(e)}")
            disconnected_clients.append(client_id)

    async def send_message(self, client_id: str, message: str) -> bool:
        """向指定客户端发送消息，返回是否发送成功"""
        client = self.client_manager.get_client(client_id)
        client_type = self.client_manager.get_client_type(client_id)
        
        if not client:
            logger.warning(f"Client {client_id} not found.")
            return False
            
        try:
            await client.send(message)
            if client_type == "expert_chat":  # 只为expert_chat类型打印消息发送成功
                logger.info(f"Message sent to {client_id}")
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected.")
            await self.unregister_client(client_id)
            return False
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {str(e)}")
            return False

    async def start(self) -> None:
        """启动WebSocket服务器"""
        try:
            self.server = await websockets.serve(self.handler, self.host, self.port)
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {str(e)}")
            raise

    async def stop(self) -> None:
        """停止WebSocket服务器"""
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                logger.info("WebSocket server stopped")
            self.client_manager.clear_all()
        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {str(e)}")


class WebSocketThread(threading.Thread):
    """WebSocket服务器线程"""
    def __init__(self):
        super().__init__(name="WebSocketThread")
        self.daemon = True  # 设置为守护线程，这样主程序退出时线程也会退出
    
    def run(self):
        asyncio.run(main())


async def shutdown() -> None:
    """优雅关闭服务器"""
    global server
    logger.info("Received shutdown signal, closing server...")
    if server:
        await server.stop()
    
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.get_running_loop().stop()


async def main() -> None:
    """主函数"""
    global server
    
    try:
        server = WebSocketServer()
        await server.start()
        while True:
            await asyncio.sleep(30)
            logger.info("WebSocket server is running")
    except asyncio.CancelledError:
        logger.info("Server task cancelled")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {str(e)}")
    finally:
        if server:
            await server.stop()


def start_server():
    """启动WebSocket服务器"""
    ws_thread = WebSocketThread()
    ws_thread.start()
    return ws_thread


# 启动服务器
thread = start_server()
 