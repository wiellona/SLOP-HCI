import http from 'http';
import app from './app';
import { Server as SocketIOServer } from 'socket.io';
import { setupSocketHandlers } from './sockets/chat.handler'; // <--- Import handler socket

const PORT = process.env.PORT || 4000;

// Bungkus aplikasi Express dengan http.Server
const server = http.createServer(app);

// Inisialisasi mesin Socket.io
const io = new SocketIOServer(server, {
    cors: {
        origin: "*", // Mengizinkan frontend dari port manapun (nanti diamankan pas produksi)
        methods: ["GET", "POST"]
    }
});

// Jalankan logika socket yang tadi kita buat
setupSocketHandlers(io);

server.listen(PORT, () => {
    console.log(`🚀 Server berjalan di http://localhost:${PORT}`);
    console.log(`⚡ Mesin Real-Time (Socket.io) aktif dan siap sedia!`);
});