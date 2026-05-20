import { Request, Response, NextFunction } from 'express';
import { prisma } from '../lib/prisma';
import { AppError } from '../middlewares/error.middleware';

const calculateLevenshtein = (a: string, b: string): number => {
    if (!a.length) return b.length;
    if (!b.length) return a.length;
    const matrix = [];
    for (let i = 0; i <= b.length; i++) { matrix[i] = [i]; }
    for (let j = 0; j <= a.length; j++) { matrix[0][j] = j; }
    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            if (b.charAt(i - 1) === a.charAt(j - 1)) {
                matrix[i][j] = matrix[i - 1][j - 1];
            } else {
                matrix[i][j] = Math.min(
                    matrix[i - 1][j - 1] + 1,
                    Math.min(matrix[i][j - 1] + 1, matrix[i - 1][j] + 1)
                );
            }
        }
    }
    return matrix[b.length][a.length];
};

export const recordCorrection = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
        const data = req.body;

        // Validasi input minimal sebelum menyentuh Prisma
        if (!data.conversation_id || !data.raw_prediction_text) {
            throw new AppError(400, 'Properti conversation_id dan raw_prediction_text wajib diisi');
        }

        let editDelta = 0;
        if (data.was_edited && data.final_corrected_text) {
            editDelta = calculateLevenshtein(data.raw_prediction_text, data.final_corrected_text);
        }

        const dummyUserId = "dummy-user-id";

        await prisma.user.upsert({
            where: { id: dummyUserId },
            update: {},
            create: {
                id: dummyUserId,
                email: "dummy.tester@slop.id",      
                password: "dummypassword123",       
                display_name: "Test User",
                role: "CUSTOMER"
            }
        });

        const log = await prisma.aI_Inference_Log.create({
            data: {
                conversation_id: data.conversation_id,
                user_id: dummyUserId,
                raw_prediction_text: data.raw_prediction_text,
                final_corrected_text: data.final_corrected_text,
                was_edited: data.was_edited,
                was_rejected: data.was_rejected,
                confidence_score: data.confidence_score,
                model_version: data.model_version,
                input_modality: data.input_modality,
                edit_delta_chars: editDelta,
                inference_latency_ms: data.inference_latency_ms,
                occlusion_detected: data.occlusion_detected,
            }
        });

        res.status(201).json({
            log_id: log.id,
            edit_delta_chars: editDelta,
            status: "recorded"
        });
    } catch (error) {
        next(error);
    }
};