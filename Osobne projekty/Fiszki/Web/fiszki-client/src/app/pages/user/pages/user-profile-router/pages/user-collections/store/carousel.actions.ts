import { createAction, props } from "@ngrx/store";

export const incrementPage = createAction(
    'incrementPage',    
)

export const decrementPage = createAction(
    'decrementPage',    
)

export const resetPage = createAction(
    'resetPage',
)

export const setCollectionQuantity = createAction(
    'setCollectionQuantity',
    props<{quantity : number}>()
)

export const setElementsToDisplay = createAction(
    'setElementsToDisplay',
    props<{value : number, pageQuantity : number}>()
)

export const changeCollectionQuantity = createAction(
    'changeCollectionQuantity',
    props<{value : number}>()
)
