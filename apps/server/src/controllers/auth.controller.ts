import { Request, Response, NextFunction } from 'express';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { prisma } from '../lib/prisma';
import { AppError } from '../middlewares/error.middleware';

// Fungsi pembantu untuk membuat JWT Token (Valid selama 1 hari)
const generateToken = (userId: string, role: string): string => {
    return jwt.sign(
        { userId, role },
        process.env.JWT_SECRET!,
        { expiresIn: '1d' }
    );
};

export const register = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
        const { id, email, password, display_name, role } = req.body;

        if (!id || !email || !password || !display_name) {
            throw new AppError(400, 'Semua kolom (id, email, password, display_name) wajib diisi');
        }

        // 1. Periksa apakah user ID atau Email sudah terdaftar
        const userExists = await prisma.user.findUnique({ where: { id } });
        if (userExists) {
            throw new AppError(400, 'User ID sudah digunakan');
        }

        // 2. Hash password menggunakan bcrypt (10 salt rounds)
        const hashedPassword = await bcrypt.hash(password, 10);

        // 3. Simpan user baru ke database Supabase via Prisma
        const newUser = await prisma.user.create({
            data: {
                id,
                email,
                password: hashedPassword,
                display_name,
                role: role || 'STAFF', // Default jika tidak diisi adalah STAFF
            }
        });

        // 4. Generate token otomatis setelah berhasil daftar
        const token = generateToken(newUser.id, newUser.role);

        res.status(201).json({
            message: 'Registrasi berhasil',
            token,
            user: {
                id: newUser.id,
                display_name: newUser.display_name,
                role: newUser.role
            }
        });
    } catch (error) {
        next(error);
    }
};

export const login = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
        const { id, password } = req.body; // Kita gunakan ID (bisa username/NIP) untuk login cepat

        if (!id || !password) {
            throw new AppError(400, 'ID dan password wajib diisi');
        }

        // 1. Cari user berdasarkan ID
        const user = await prisma.user.findUnique({ where: { id } });
        if (!user || !user.password) {
            throw new AppError(401, 'ID atau password salah');
        }

        // 2. Cocokkan password input dengan password ter-hash di database
        const isPasswordMatch = await bcrypt.compare(password, user.password);
        if (!isPasswordMatch) {
            throw new AppError(401, 'ID atau password salah');
        }

        // 3. Buat token JWT baru yang sah
        const token = generateToken(user.id, user.role);

        res.status(200).json({
            message: 'Login berhasil',
            token,
            user: {
                id: user.id,
                display_name: user.display_name,
                role: user.role
            }
        });
    } catch (error) {
        next(error);
    }
};