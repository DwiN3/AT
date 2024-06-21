import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BlogItemImageComponent } from './blog-item-image.component';

describe('BlogItemImageComponent', () => {
  let component: BlogItemImageComponent;
  let fixture: ComponentFixture<BlogItemImageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BlogItemImageComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(BlogItemImageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
