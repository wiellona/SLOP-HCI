import { Request } from 'express';

export interface TokenPayload {
    userId: string;
    role: 'STAFF' | 'CUSTOMER' | 'ADMIN';
}

// Interface khusus agar req.user dikenali oleh TypeScript
export interface AuthenticatedRequest extends Request {
    user?: {
        id: string;
        display_name: string;
        role: 'STAFF' | 'CUSTOMER' | 'ADMIN';
    };
}