import { Router } from 'express';
import { recordCorrection } from '../controllers/rlhf.controller';

const router = Router();

// POST /api/v1/rlhf/corrections
router.post('/corrections', recordCorrection);

export default router;