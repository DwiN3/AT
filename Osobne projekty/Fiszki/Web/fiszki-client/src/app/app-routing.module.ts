import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { ErrorPageComponent } from './pages/error-page/error-page.component';
import { HomeComponent } from './pages/home/home.component';
import { HomeGuard } from './pages/home/services/home-guard';
import { PasswordResetComponent } from './pages/password-reset/password-reset.component';
import { RegisterComponent } from './pages/register/register.component';
import { UserRoutingModule } from './pages/user/user-routing.module';

const routes : Routes = 
[
  { path: '', component : HomeComponent, canActivate : [HomeGuard] },
  { path : 'register', component : RegisterComponent, canActivate : [HomeGuard]},
  { path : 'password-reset', component : PasswordResetComponent, canActivate : [HomeGuard]},
  {
    path: 'user',
    loadChildren: () => import('./pages/user/user.module').then((m) => m.UserModule),
  },
  { path: '**', component: ErrorPageComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes), UserRoutingModule],
  exports: [RouterModule]
})
export class AppRoutingModule { }
