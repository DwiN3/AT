import { Routes } from '@angular/router';
import { authGuard } from './services/auth/auth.guard';
import { guestGuard } from './services/guest/guest.guard';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./components/home/home.component').then(m => m.HomeComponent)
  },
  {
    path: 'login',
    loadComponent: () => import('./components/login/login.component').then(m => m.LoginComponent),
    canActivate: [guestGuard]
  },
  {
    path: 'signup',
    loadComponent: () => import('./components/signup/signup.component').then(m => m.SignupComponent),
    canActivate: [guestGuard]
  },
  {
    path: 'blog',
    loadComponent: () => import('./components/blog-components/blog-home/blog-home.component').then(m => m.BlogHomeComponent),
    canActivate: [authGuard]
  },
  {
    path: 'blog/detail/:id',
    loadComponent: () => import('./components/blog-components/blog-item-details/blog-item-details.component').then(m => m.BlogItemDetailsComponent),
    canActivate: [authGuard]
  },
  {
    path: 'create-post',
    loadComponent: () => import('./components/create-post/create-post.component').then(m => m.CreatePostComponent),
    canActivate: [authGuard]
  },
  {
    path: 'account',
    loadComponent: () => import('./components/account/account.component').then(m => m.AccountComponent),
    canActivate: [authGuard]
  },
  {
    path: '**', 
    redirectTo: '', 
    pathMatch: 'full'
  }
];