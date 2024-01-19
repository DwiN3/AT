import { createReducer, on } from "@ngrx/store";
import { setCollection, setLanguage, setLearningMode } from "./game-settings.action";
import { initialState } from "./game.state";

export const gameSettingFeautureKey = 'gameSettings';

const _gameSettingsReducer = createReducer(
    initialState,
    on(setLearningMode, (state, action) => {
        return{
            ...state,
            learningMode : action.mode 
        }
    }),
    on(setLanguage, (state) => {
        return{
            ...state,
            polishFirst : !state.polishFirst,
        };
    }),
    on(setCollection, (state, action) => {
        return{
            ...state,
            flashcards : action.collection
        }
    })
);

export function gameSettingsReducer(state : any, action : any){
    return _gameSettingsReducer(state, action);
}