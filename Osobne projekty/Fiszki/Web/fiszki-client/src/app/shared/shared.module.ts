import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AlertComponent } from './ui/alert/alert.component';
import { LoaderComponent } from './ui/loader/loader.component';
import { LogoComponent } from './ui/logo/logo.component';
import { PlaceholderDirective } from './ui/alert/directive/placeholder.directive';

@NgModule({
  declarations: [
    AlertComponent,
    LoaderComponent,
    LogoComponent,
    PlaceholderDirective,
  ],
  imports: [
    CommonModule,
  ],
  exports: [
    AlertComponent,
    LoaderComponent,
    LogoComponent,
    PlaceholderDirective,
  ],
})
export class SharedModule {}