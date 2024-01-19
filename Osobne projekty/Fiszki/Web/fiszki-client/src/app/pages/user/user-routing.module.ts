import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AuthGuard } from 'src/app/shared/services/auth-guard.service';
import { ErrorPageComponent } from '../error-page/error-page.component';
import { LearningPageComponent } from './pages/learning-page/learning-page.component';
import { CategoriesPageComponent } from './pages/learning-page/pages/categories-page/categories-page.component';
import { CollectionPageComponent } from './pages/learning-page/pages/collection-page/collection-page.component';
import { GamePageComponent } from './pages/learning-page/pages/game-page/game-page.component';
import { LearningPageSettingsComponent } from './pages/learning-page/pages/learning-page-settings/learning-page-settings.component';
import { UserHomeComponent } from './pages/user-home/user-home.component';
import { UserChangePasswordComponent } from './pages/user-profile-router/pages/user-change-password/user-change-password.component';
import { UserCollectionEditComponent } from './pages/user-profile-router/pages/user-collection-edit/user-collection-edit.component';
import { CollectionRouterOutletComponent } from './pages/user-profile-router/pages/user-collections/components/collection-router-outlet/collection-router-outlet.component';
import { UserCollectionsComponent } from './pages/user-profile-router/pages/user-collections/user-collections.component';
import { UserProfileComponent } from './pages/user-profile-router/pages/user-profile/user-profile.component';
import { UserProfileRouterComponent } from './pages/user-profile-router/user-profile-router.component';
import { UserLevelResolver } from './services/user-level-resolver';
import { UserComponent } from './user.component';

const routes : Routes = 
[
    { 
        path: 'user',
        canActivate : [AuthGuard], 
        component: UserComponent, 
        children: [
            {
                path: '',
                component: UserHomeComponent,
            },
            {
                path: 'profile',
                component: UserProfileRouterComponent,
                children: 
                [
                    {
                        path: '',
                        resolve : { levelInfo : UserLevelResolver},
                        component: UserProfileComponent,
                    },
                    {
                        path: 'change-password',
                        resolve : { levelInfo : UserLevelResolver},
                        component: UserChangePasswordComponent,
                    },
                    {
                        path: 'collections',
                        component: CollectionRouterOutletComponent,
                        children: 
                        [
                            {
                                path: '',
                                component: UserCollectionsComponent,
                            },
                            {
                                path: 'edit',
                                component: UserCollectionEditComponent,
                            }
                        ]

                    }
                ]
            },
            {
                path: 'learning',
                component: LearningPageComponent,
                children:
                [
                    {
                        path: '',
                        component: LearningPageSettingsComponent,
                    },
                    {
                        path: 'categories',
                        component: CategoriesPageComponent,
                    },
                    {
                        path : 'game',
                        component: GamePageComponent,
                    },
                    {
                        path : 'collections',
                        component: CollectionPageComponent,
                    },
                ]
            },
            { path : '**', component : ErrorPageComponent},
        ] 
    },
    
]

@NgModule({
    imports: [RouterModule.forChild(routes)],
    exports: [RouterModule]
})

export class UserRoutingModule{}