import { Request, Response, NextFunction } from 'express';
import { prisma } from '../lib/prisma';
import { AppError } from '../middlewares/error.middleware';
import { AuthenticatedRequest } from '../types/auth';

export const createSession = async (
    req: AuthenticatedRequest, 
    res: Response,
    next: NextFunction
): Promise<void> => {
    try {
        // Menggunakan ID asli dari token JWT hasil ekstraksi satpam protect
        if (!req.user) {
            throw new AppError(401, 'Autentikasi diperlukan');
        }

        const staffId = req.user.id;
        const staffRole = req.user.role;

        // Memastikan hanya STAFF atau ADMIN yang bisa men-generate sesi baru
        if (staffRole === 'CUSTOMER') {
            throw new AppError(403, 'Hanya Staff yang dapat membuat sesi percakapan baru');
        }

        const newSession = await prisma.conversation.create({
            data: {
                status: 'ACTIVE',
            },
        });

        res.status(201).json({
            message: 'Sesi percakapan berhasil dibuat',
            session_token: newSession.session_token,
            session_id: newSession.id,
        });
    } catch (error) {
        next(error);
    }
};

export const getAllSessions = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
        const sessions = await prisma.conversation.findMany({
            orderBy: { started_at: 'desc' },
        });
        res.status(200).json(sessions);
    } catch (error) {
        next(error);
    }
};

export const getSession = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
        const { id } = req.params as { id: string };
        const session = await prisma.conversation.findUnique({
            where: { id },
            include: { participants: true },
        });

        if (!session) {
            throw new AppError(404, 'Sesi percakapan tidak ditemukan');
        }

        res.status(200).json(session);
    } catch (error) {
        next(error);
    }
};

export const endSession = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
        const { id } = req.params as { id: string };

        // Validasi apakah sesi ada sebelum melakukan update
        const sessionExists = await prisma.conversation.findUnique({ where: { id } });
        if (!sessionExists) {
            throw new AppError(404, 'Sesi tidak ditemukan atau gagal diakhiri');
        }

        const updatedSession = await prisma.conversation.update({
            where: { id },
            data: {
                status: 'COMPLETED',
                ended_at: new Date()
            }
        });

        res.status(200).json({ message: 'Sesi diakhiri', session: updatedSession });
    } catch (error) {
        next(error);
    }
};

export const getSessionMessages = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
        const { id } = req.params as { id: string };

        // Validasi apakah sesi ada sebelum menarik pesan
        const sessionExists = await prisma.conversation.findUnique({ where: { id } });
        if (!sessionExists) {
            throw new AppError(404, 'Tidak dapat mengambil pesan, sesi tidak ditemukan');
        }

        const messages = await prisma.message.findMany({
            where: { conversation_id: id },
            orderBy: { created_at: 'asc' }
        });

        res.status(200).json(messages);
    } catch (error) {
        next(error);
    }
};