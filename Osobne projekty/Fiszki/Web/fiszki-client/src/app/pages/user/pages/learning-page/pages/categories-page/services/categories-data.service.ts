import { Injectable } from "@angular/core";
import { CategoryModel } from "../models/category.model";

import { faFootball, faBurger, faMasksTheater, faHouse, faCompass, faBriefcase } from "@fortawesome/free-solid-svg-icons";

@Injectable({ providedIn : 'root' })
export class CategoriesDataService
{
    categories : CategoryModel[] = 
    [
        {
            categoryName : 'sport',
            icon : faFootball,
        },
        {
            categoryName : 'jedzenie',
            icon : faBurger,
        },
        {
            categoryName : 'sztuka',
            icon : faMasksTheater,
        },
        {
            categoryName : 'dom',
            icon : faHouse,
        },
        {
            categoryName : 'podróże',
            icon : faCompass,
        },
        {
            categoryName : 'praca',
            icon : faBriefcase,
        },
    ];


}