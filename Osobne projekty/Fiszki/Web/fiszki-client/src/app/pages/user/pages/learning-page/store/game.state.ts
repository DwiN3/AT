import { BaseFlashcardInterface } from "src/app/shared/models/flashcard.interface";

export interface GameSettingsState
{
    learningMode : string;
    polishFirst : boolean;
    flashcards : BaseFlashcardInterface[];
}

export const initialState = 
{
    learningMode : 'learning',
    polishFirst : true,
    flashcards : [],
}