import { Server, Socket } from 'socket.io';
import { prisma } from '../lib/prisma';

export const setupSocketHandlers = (io: Server) => {
    io.on('connection', (socket: Socket) => {
        console.log(`🟢 Klien terhubung: ${socket.id}`);

        // 1. Klien (Staff/Customer) bergabung ke room menggunakan session_token
        socket.on('join_session', ({ sessionToken, role }) => {
            socket.join(sessionToken);
            console.log(`👥 Klien ${socket.id} (${role}) masuk ke room: ${sessionToken}`);

            // Beri tahu layar sebelah kalau ada yang bergabung
            socket.to(sessionToken).emit('user_joined', { role, message: `${role} telah bergabung.` });
        });

        // 2. Menerima pesan baru dan menyebarkannya ke layar sebelah
        socket.on('send_message', async (data) => {
            /* Ekspektasi data dari frontend: 
            { sessionToken, senderId, content, modality } 
            */
            try {
                // Cari ID sesi berdasarkan token
                const conversation = await prisma.conversation.findUnique({
                    where: { session_token: data.sessionToken }
                });

                if (!conversation) return;

                // Pastikan dummy user untuk sender_id ada (buat testing aja)
                await prisma.user.upsert({
                    where: { id: data.senderId },
                    update: {},
                    create: {
                        id: data.senderId,
                        email: `${data.senderId}@slop.id`,  // <--- Tambahkan ini (dinamis sesuai ID sender)
                        password: "dummypassword123",       // <--- Tambahkan ini
                        display_name: "Pengirim",
                        role: "CUSTOMER"
                    }
                });

                // Simpan pesan permanen ke database
                const newMessage = await prisma.message.create({
                    data: {
                        conversation_id: conversation.id,
                        sender_id: data.senderId,
                        content: data.content,
                        modality: data.modality,
                        status: 'SENT',
                        sent_at: new Date()
                    }
                });

                // 🚀 Tembakkan pesan ini ke SEMUA KLIEN yang ada di room tersebut
                io.to(data.sessionToken).emit('new_message', newMessage);

            } catch (error) {
                console.error('❌ Gagal memproses pesan socket:', error);
            }
        });

        socket.on('disconnect', () => {
            console.log(`🔴 Klien terputus: ${socket.id}`);
        });
    });
};