import { Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { prisma } from '../lib/prisma';
import { AppError } from './error.middleware';
import { AuthenticatedRequest, TokenPayload } from '../types/auth';

export const protect = async (
    req: AuthenticatedRequest,
    res: Response,
    next: NextFunction
): Promise<void> => {
    try {
        let token: string | undefined;

        // 1. Periksa token di header Authorization (Format: Bearer <token>)
        if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
            token = req.headers.authorization.split(' ')[1];
        }

        if (!token) {
            throw new AppError(401, 'Anda belum login, token tidak ditemukan');
        }

        // 2. Verifikasi token menggunakan JWT_SECRET
        const decoded = jwt.verify(token, process.env.JWT_SECRET!) as TokenPayload;

        // 3. Cari user di database Supabase berdasarkan userId dari token
        const currentUser = await prisma.user.findUnique({
            where: { id: decoded.userId },
            select: { id: true, display_name: true, role: true } // Ambil data yang diperlukan saja
        });

        if (!currentUser) {
            throw new AppError(401, 'User pemilik token ini sudah tidak terdaftar');
        }

        // 4. Tempelkan data user yang sah ke objek request
        req.user = currentUser;
        next();
    } catch (error: any) {
        // Tangani jika token kedaluwarsa atau diubah secara ilegal
        if (error.name === 'JsonWebTokenError') {
            next(new AppError(401, 'Token tidak valid, silakan login ulang'));
        } else if (error.name === 'TokenExpiredError') {
            next(new AppError(401, 'Token sudah kedaluwarsa, silakan login ulang'));
        } else {
            next(error);
        }
    }
};