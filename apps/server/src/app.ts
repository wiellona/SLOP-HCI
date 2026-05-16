import express, { Application, Request, Response } from 'express';
import cors from 'cors';
import sessionRoutes from './routes/session.route';
import rlhfRoutes from './routes/rlhf.route';
import { errorHandler } from './middlewares/error.middleware'
import authRoutes from './routes/auth.route';

const app: Application = express();

app.use(cors());
app.use(express.json());

app.get('/api/v1/health', (req: Request, res: Response) => {
    res.status(200).json({ status: 'ok', message: 'SLOP Server is running' });
});

app.use('/api/v1/sessions', sessionRoutes);
app.use('/api/v1/rlhf', rlhfRoutes);
app.use('/api/v1/auth', authRoutes);

// Middleware penanganan error global
app.use(errorHandler);

export default app;