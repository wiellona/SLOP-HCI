import { PrismaClient } from '@prisma/client';
import { Pool } from 'pg';
import { PrismaPg } from '@prisma/adapter-pg';
import dotenv from 'dotenv';

dotenv.config();

// Mencegah multiple instances saat hot-reload
const globalForPrisma = global as unknown as { prisma: PrismaClient };

// Buat koneksi Pool standar ke Supabase
const pool = new Pool({
    connectionString: process.env.DATABASE_URL!,
});

// Aadapter khusus Prisma untuk pg
const adapter = new PrismaPg(pool);

// Inject adapter ke dalam PrismaClient
export const prisma =
    globalForPrisma.prisma ||
    new PrismaClient({
        adapter,
        log: ['query', 'error', 'warn'],
    });

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma;