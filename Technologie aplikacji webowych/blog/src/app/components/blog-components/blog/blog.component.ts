import { Component, OnInit, Input } from '@angular/core';
import { CommonModule } from "@angular/common";
import { DataService } from "../../../services/post/post.service";
import { BlogItemComponent } from "../blog-item/blog-item.component";
import { SearchBarComponent } from "../../search-bar/search-bar.component";
import { FilterTextPipe } from "../../../pipes/filter-text/filter-text.pipe";

@Component({
  selector: 'blog',
  standalone: true,
  imports: [ BlogItemComponent, CommonModule, SearchBarComponent, FilterTextPipe ],
  providers: [DataService],
  templateUrl: './blog.component.html',
  styleUrl: './blog.component.css'
})
export class BlogComponent implements OnInit {

  @Input() filterText: string = '';
  public items$: any;

  constructor(private service: DataService) { }

  ngOnInit() {
    this.getAll();
  }

  getAll(){
    this.service.getAll().subscribe(response => {
      this.items$ = response;
    });
  }
}

