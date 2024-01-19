import { CommonModule } from "@angular/common";
import { NgModule } from "@angular/core";
import { UserRoutingModule } from "./user-routing.module";
import { UserComponent } from "./user.component";
import { NavbarComponent } from './core/navbar/navbar.component';
import { UserHomeComponent } from "./pages/user-home/user-home.component";
import { AuthGuard } from "src/app/shared/services/auth-guard.service";
import { SharedModule } from "src/app/shared/shared.module";
import { UserProfileComponent } from './pages/user-profile-router/pages/user-profile/user-profile.component';
import { UserLevelResolver } from "./services/user-level-resolver";
import { LevelDirective } from "./pages/user-profile-router/pages/user-profile/directives/level-status";
import { UserProfileRouterComponent } from './pages/user-profile-router/user-profile-router.component';
import { UserCollectionsComponent } from './pages/user-profile-router/pages/user-collections/user-collections.component';
import { StoreModule } from "@ngrx/store";
import { carouselFeatureKey, carouselReducer } from "./pages/user-profile-router/pages/user-collections/store/carousel.reducer";
import { FontAwesomeModule } from "@fortawesome/angular-fontawesome";
import { CollectionComponent } from './pages/user-profile-router/pages/user-collections/components/collection/collection.component';
import { CarouselDirective } from "./pages/user-profile-router/pages/user-collections/directives/carousel.directive";
import { LearningPageComponent } from './pages/learning-page/learning-page.component';
import { LearningPageSettingsComponent } from './pages/learning-page/pages/learning-page-settings/learning-page-settings.component';
import { FormsModule } from "@angular/forms";
import { gameSettingFeautureKey, gameSettingsReducer } from "./pages/learning-page/store/game-settings.reducer";
import { CategoriesPageComponent } from './pages/learning-page/pages/categories-page/categories-page.component';
import { CategoriesItemComponent } from './pages/learning-page/pages/categories-page/components/categories-item/categories-item.component';
import { GamePageComponent } from './pages/learning-page/pages/game-page/game-page.component';
import { LearningModeComponent } from './pages/learning-page/pages/game-page/components/learning-mode/learning-mode.component';
import { AddCollectionComponent } from './pages/user-profile-router/pages/user-collections/components/add-collection/add-collection.component';
import { collectionsFeatureKey, collectionsReducer } from "./pages/user-profile-router/pages/user-collections/store/collections.reducer";
import { UserCollectionEditComponent } from './pages/user-profile-router/pages/user-collection-edit/user-collection-edit.component';
import { CollectionRouterOutletComponent } from './pages/user-profile-router/pages/user-collections/components/collection-router-outlet/collection-router-outlet.component';
import { FlashcardComponent } from './pages/user-profile-router/pages/user-collections/components/flashcard/flashcard.component';
import { FlashcardEditFormComponent } from './pages/user-profile-router/pages/user-collection-edit/components/flashcard-edit-form/flashcard-edit-form.component';
import { SummaryBoxComponent } from './pages/learning-page/pages/game-page/components/summary-box/summary-box.component';
import { CollectionPageComponent } from './pages/learning-page/pages/collection-page/collection-page.component';
import { CollectionItemComponent } from "./pages/learning-page/pages/collection-page/components/collection-item/collection-item.component";
import { QuizModeComponent } from './pages/learning-page/pages/game-page/components/quiz-mode/quiz-mode.component';
import { QuizSummaryComponent } from './pages/learning-page/pages/game-page/components/quiz-mode/quiz-summary/quiz-summary.component';
import { UserChangePasswordComponent } from './pages/user-profile-router/pages/user-change-password/user-change-password.component';
import { FooterComponent } from './core/footer/footer.component';

@NgModule({
    declarations:[
        NavbarComponent,
        UserComponent,
        UserHomeComponent,
        UserProfileComponent,
        LevelDirective,
        UserProfileRouterComponent,
        UserCollectionsComponent,
        CollectionComponent,
        CarouselDirective,
        LearningPageComponent,
        LearningPageSettingsComponent,
        CategoriesPageComponent,
        CategoriesItemComponent,
        GamePageComponent,
        LearningModeComponent,
        AddCollectionComponent,
        UserCollectionEditComponent,
        CollectionRouterOutletComponent,
        FlashcardComponent,
        FlashcardEditFormComponent,
        SummaryBoxComponent,
        CollectionPageComponent,
        CollectionItemComponent,
        QuizModeComponent,
        QuizSummaryComponent,
        UserChangePasswordComponent,
        FooterComponent,
    ],
    imports: [
        CommonModule,
        UserRoutingModule,
        SharedModule,
        StoreModule.forFeature(carouselFeatureKey, carouselReducer),
        StoreModule.forFeature(gameSettingFeautureKey, gameSettingsReducer),
        StoreModule.forFeature(collectionsFeatureKey, collectionsReducer),
        FontAwesomeModule,
        FormsModule,
    ],
    exports: [
        CarouselDirective,
    ],
    providers: [
        AuthGuard,
        UserComponent,
        UserLevelResolver,
    ]
})

export class UserModule {}