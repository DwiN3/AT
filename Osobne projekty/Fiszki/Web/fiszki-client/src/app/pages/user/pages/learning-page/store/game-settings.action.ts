import { createAction, props } from "@ngrx/store";
import { BaseFlashcardInterface } from "src/app/shared/models/flashcard.interface";

export const setLearningMode = createAction(
    'setLearningMode',
    props<{mode : string}>()
)

export const setLanguage = createAction(
    'setLanguage'
)

export const setCollection = createAction(
    'setCollection',
    props<{collection : any}>()
)
