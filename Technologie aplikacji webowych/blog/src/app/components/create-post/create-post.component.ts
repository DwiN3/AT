import { Component, Renderer2 } from '@angular/core';
import { FormsModule } from "@angular/forms";
import { Router } from "@angular/router";
import { ChangeDetectorRef } from '@angular/core';
import { DataService } from "../../services/post/post.service";

@Component({
  selector: 'app-create-post',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './create-post.component.html',
  styleUrls: ['./create-post.component.css']
})
export class CreatePostComponent {
  post = {
    title: '',
    image: '',
    text: ''
  };

  public imageUrl: string | undefined;

  constructor(
    private dataService: DataService, 
    private router: Router) {}

  addPost() {
    this.dataService.addPost(this.post).subscribe(response => {
      this.router.navigate(['/blog']);
    }, error => {
      console.error('Error adding post', error);
    });
  }

  displayImage() {
    this.imageUrl = this.post.image;
  }
}
