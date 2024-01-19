import { createReducer, on } from "@ngrx/store";
import { initialState } from './collections.state'
import { addCollection, deleteCollection, setCollection } from "./collections.actions";

export const collectionsFeatureKey = 'collections'

const _collectionsReducer = createReducer(
    initialState,
    on(setCollection, (state, action) => {
        return{
            ...state,
            collections : action.collections
        }
    }),
    on(addCollection, (state, action) => {
        return{
            ...state,
            collections : state.collections.concat(action.collection)
        }
    }),
    on(deleteCollection, (state, action) => {
        return{
            ...state,
            collections : state.collections.filter(collection => collection.collectionName !== action.name)
        }
    }),
)

export function collectionsReducer(state : any, action : any){
    return _collectionsReducer(state, action);
}