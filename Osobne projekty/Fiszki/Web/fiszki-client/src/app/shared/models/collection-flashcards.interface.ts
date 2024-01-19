import { BaseFlashcardInterface } from "./flashcard.interface";

export interface CollectionFlashcardsInterface
{
    collectionName : string;
    flashcards : BaseFlashcardInterface[];
}