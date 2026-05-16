import { Router } from 'express';
import {
    createSession,
    getAllSessions,
    getSession,
    endSession,
    getSessionMessages
} from '../controllers/session.controller';
import { protect } from '../middlewares/auth.middleware';

const router = Router();

router.post('/', protect, createSession);
router.get('/', protect, getAllSessions);
router.get('/:id', protect, getSession);
router.patch('/:id/end', protect, endSession);
router.get('/:id/messages', protect, getSessionMessages);

export default router;