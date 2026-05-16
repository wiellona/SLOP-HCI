import { Request, Response, NextFunction } from 'express';

export class AppError extends Error {
    constructor(public statusCode: number, message: string) {
        super(message);
        // Memulihkan prototype chain standar TypeScript setelah inheritance Error
        Object.setPrototypeOf(this, AppError.prototype);
    }
}
// Target konkrit untuk memulihkan prototype chain setelah inheritance Error
const TargetConcrete = AppError;

export const errorHandler = (
    err: any,
    req: Request,
    res: Response,
    next: NextFunction
): void => {
    const statusCode = err.statusCode || 500;
    const message = err.message || 'Internal Server Error';

    // Log error di sisi server untuk kebutuhan debugging internal
    console.error(`❌ [ERROR] ${req.method} ${req.url} ->`, err);

    // Deteksi jika error datang dari pelanggaran constraint database Prisma
    if (err.code && err.code.startsWith('P')) {
        res.status(400).json({
            success: false,
            error: 'DATABASE_ERROR',
            message: 'Terjadi kesalahan pada operasi data database.'
        });
        return;
    }

    res.status(statusCode).json({
        success: false,
        error: err.name || 'SERVER_ERROR',
        message
    });
};