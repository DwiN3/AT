import { Component, Input, OnInit } from '@angular/core';
import { CommonModule } from "@angular/common";
import { BlogItemImageComponent } from "../blog-item-image/blog-item-image.component";
import { BlogItemTextComponent } from "../blog-item-text/blog-item-text.component";
import { DataService } from "../../../services/post/post.service";
import { AuthService } from '../../../services/auth/auth.service';

@Component({
  selector: 'blog-item',
  standalone: true,
  imports: [BlogItemImageComponent, BlogItemTextComponent, CommonModule],
  providers: [DataService],
  templateUrl: './blog-item.component.html',
  styleUrls: ['./blog-item.component.css']
})
export class BlogItemComponent implements OnInit {
  @Input() title?: string;
  @Input() image?: string;
  @Input() text?: string;
  @Input() id?: string;

  public isAdmin?: boolean;

  constructor(
    private dataService: DataService,
    private authService: AuthService) { }

  ngOnInit(): void {
    this.isAdmin = this.authService.isAdmin();
  }

  removePost() {
    if (this.id != null && this.isAdmin) {
      this.dataService.removePost(this.id).subscribe(response => {
      }, error => {
        console.error("Error removing post:", error);
      });
      window.location.reload();
    } 
  }
}
