package com.example.Fiszki.flashcards;

import com.example.Fiszki.flashcards.flashcard.ApiError;
import com.example.Fiszki.flashcards.request.CollectionAddRequest;
import com.example.Fiszki.flashcards.request.FlashcardAddRequest;
import com.example.Fiszki.flashcards.request.FlashcardCategoryLimitRequest;
import com.example.Fiszki.flashcards.response.FlashcardCollectionResponse;
import com.example.Fiszki.flashcards.response.FlashcardInfoResponse;
import com.example.Fiszki.flashcards.response.FlashcardReturnResponse;
import com.example.Fiszki.security.auth.response.OtherException;
import com.example.Fiszki.security.auth.response.UserInfoResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.Map;

/**
 * Controller for handling flashcard-related operations.
 */
@RestController
@RequestMapping("/flashcards/")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:4200")
public class FlashcardController {
    private final FlashcardService flashcardService;

    /**
     * Endpoint for adding a new flashcard.
     *
     * @param request The request containing flashcard information.
     * @return ResponseEntity containing the added flashcard or an error response.
     */
    @PostMapping("/add-flashcard")
    public ResponseEntity<?> addFlashcard(@RequestBody FlashcardAddRequest request) {
        try {
            FlashcardReturnResponse response = flashcardService.addFlashcard(request);
            return ResponseEntity.ok(response);
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(FlashcardInfoResponse.builder().response(e.getMessage()).build());
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(FlashcardInfoResponse.builder().response(e.getMessage()).build());
        }
    }

    /**
     * Endpoint for editing an existing flashcard.
     *
     * @param flashcardsId The ID of the flashcard to be edited.
     * @param request      The request containing updated flashcard information.
     * @return ResponseEntity containing the edit response or an error response.
     */
    @PutMapping("/edit/{flashcardsId}")
    public ResponseEntity<FlashcardInfoResponse> editFlashcard(@PathVariable Integer flashcardsId, @RequestBody FlashcardAddRequest request) {
        try {
            FlashcardInfoResponse response = flashcardService.editFlashcard(flashcardsId, request);
            return ResponseEntity.ok(response);
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(FlashcardInfoResponse.builder().response(e.getMessage()).build());
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(FlashcardInfoResponse.builder().response( e.getMessage()).build());
        }
    }

    /**
     * Endpoint for showing a flashcard by its ID.
     *
     * @param flashcardsId The ID of the flashcard to be retrieved.
     * @return ResponseEntity containing the flashcard or an error response.
     */
    @GetMapping("/show/{flashcardsId}")
    public ResponseEntity<?> showFlashcard(@PathVariable Integer flashcardsId) {
        try {
            FlashcardReturnResponse response = flashcardService.showFlashcardById(flashcardsId);
            if (response == null) {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(UserInfoResponse.builder().response("Flashcard with given id does not exist.").build());
            }
            return ResponseEntity.ok(response);
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(FlashcardInfoResponse.builder().response(e.getMessage()).build());
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(FlashcardInfoResponse.builder().response(e.getMessage()).build());
        }
    }

    /**
     * Endpoint for deleting a flashcard by its ID.
     *
     * @param flashcardId The ID of the flashcard to be deleted.
     * @return ResponseEntity containing the delete response or an error response.
     */
    @DeleteMapping("/delete/{flashcardId}")
    public ResponseEntity<FlashcardInfoResponse> deleteFlashcardById(@PathVariable Integer flashcardId) {
        try {
            FlashcardInfoResponse response = flashcardService.deleteFlashcardById(flashcardId);
            return ResponseEntity.ok(response);
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(FlashcardInfoResponse.builder().response(e.getMessage()).build());
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(FlashcardInfoResponse.builder().response(e.getMessage()).build());
        }
    }

    /**
     * Endpoint for showing flashcards by category.
     *
     * @param category The category of flashcards to retrieve.
     * @return ResponseEntity containing the list of flashcards or an error response.
     */
    @GetMapping("/category/{category}")
    public ResponseEntity<?> showFlashcardsByCategory(@PathVariable String category) {
        try {
            List<FlashcardReturnResponse> response = flashcardService.showFlashcardsByCategory(category);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            ApiError apiError = new ApiError(HttpStatus.INTERNAL_SERVER_ERROR, "Internal server error occurred", e.getMessage());
            return ResponseEntity.status(apiError.getStatus()).body(apiError);
        }
    }

    /**
     * Endpoint for showing flashcards by category with a limit.
     *
     * @param request  The request containing the category limit.
     * @param category The category of flashcards to retrieve.
     * @return ResponseEntity containing the list of flashcards or an error response.
     */
    @PostMapping("/category-limit/{category}")
    public ResponseEntity<?> showFlashcardsByCategoryWithLimit(@RequestBody FlashcardCategoryLimitRequest request, @PathVariable String category) {
        try {
            List<FlashcardReturnResponse> response = flashcardService.showFlashcardsByCategoryWithLimit(request, category);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            ApiError apiError = new ApiError(HttpStatus.INTERNAL_SERVER_ERROR, "Internal server error occurred", e.getMessage());
            return ResponseEntity.status(apiError.getStatus()).body(apiError);
        }
    }

    /**
     * Endpoint for showing all flashcard collections.
     *
     * @return ResponseEntity containing the list of flashcard collections or an error response.
     */
    @GetMapping("/collections")
    public ResponseEntity<?> showAllCollection() {
        try {
            List<FlashcardCollectionResponse> collections = flashcardService.showAllCollection();
            return ResponseEntity.ok(collections);
        } catch (Exception e) {
            ApiError apiError = new ApiError(HttpStatus.INTERNAL_SERVER_ERROR, "Internal server error occurred", e.getMessage());
            return ResponseEntity.status(apiError.getStatus()).body(apiError);
        }
    }

    /**
     * Endpoint for showing information about all flashcard collections.
     *
     * @return ResponseEntity containing the list of collection information or an error response.
     */
    @GetMapping("/collections-info")
    public ResponseEntity<?> showCollectionInfo() {
        try {
            List<Map<String, Object>> collections = flashcardService.showCollectionInfo();
            return ResponseEntity.ok(collections);
        } catch (Exception e) {
            ApiError apiError = new ApiError(HttpStatus.INTERNAL_SERVER_ERROR, "Internal server error occurred", e.getMessage());
            return ResponseEntity.status(apiError.getStatus()).body(apiError);
        }
    }

    /**
     * Endpoint for showing flashcards in a specific collection by name.
     *
     * @param nameCollection The name of the flashcard collection.
     * @return ResponseEntity containing the list of flashcards or an error response.
     */
    @GetMapping("/collection/{nameCollection}")
    public ResponseEntity<?> showCollectionByName(@PathVariable String nameCollection) {
        try {
            List<FlashcardReturnResponse> collections = flashcardService.showCollectionByName(nameCollection);
            return ResponseEntity.ok(collections);
        } catch (Exception e) {
            ApiError apiError = new ApiError(HttpStatus.INTERNAL_SERVER_ERROR, "Internal server error occurred", e.getMessage());
            return ResponseEntity.status(apiError.getStatus()).body(apiError);
        }
    }

    /**
     * Endpoint for deleting a flashcard collection by name.
     *
     * @param nameCollection The name of the flashcard collection to be deleted.
     * @return ResponseEntity containing the delete response or an error response.
     */
    @DeleteMapping("/collection/{nameCollection}")
    public ResponseEntity<FlashcardInfoResponse> deleteCollectionByName(@PathVariable String nameCollection) {
        try {
            FlashcardInfoResponse response = flashcardService.deleteCollectionByName(nameCollection);
            return ResponseEntity.ok(response);
        } catch (OtherException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(FlashcardInfoResponse.builder().response(e.getMessage()).build());
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(FlashcardInfoResponse.builder().response(e.getMessage()).build());
        }
    }

    /**
     * Endpoint for adding a new flashcard collection.
     *
     * @param request The request containing the collection name.
     * @return ResponseEntity containing the added collection or an error response.
     */
    @PostMapping("/add_collection")
    public ResponseEntity<FlashcardInfoResponse> addCollection(@RequestBody CollectionAddRequest request) {
        try {

            FlashcardInfoResponse response = flashcardService.addCollection(request);
            return ResponseEntity.ok(response);
        } catch (OtherException e) {
            return ResponseEntity.badRequest().body(FlashcardInfoResponse.builder().response(e.getMessage()).build());
        }
    }
}