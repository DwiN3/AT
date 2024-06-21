import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { BlogComponent } from "./components/blog-components/blog/blog.component";
import { NavbarComponent } from "./components/navbar/navbar.component";

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, BlogComponent, NavbarComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {}