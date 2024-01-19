package com.example.Fiszki.flashcards;

import com.example.Fiszki.Instance.TokenInstance;
import com.example.Fiszki.flashcards.flashcard.CollectionRepository;
import com.example.Fiszki.flashcards.flashcard.Flashcard;
import com.example.Fiszki.flashcards.flashcard.FlashcardCollection;
import com.example.Fiszki.flashcards.flashcard.FlashcardRepository;
import com.example.Fiszki.flashcards.request.CollectionAddRequest;
import com.example.Fiszki.flashcards.request.FlashcardAddRequest;
import com.example.Fiszki.flashcards.request.FlashcardCategoryLimitRequest;
import com.example.Fiszki.flashcards.response.FlashcardCollectionResponse;
import com.example.Fiszki.flashcards.response.FlashcardInfoResponse;
import com.example.Fiszki.flashcards.response.FlashcardReturnResponse;
import com.example.Fiszki.security.auth.response.OtherException;
import org.springframework.stereotype.Service;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Service class for handling flashcard-related operations.
 */
@Service
public class FlashcardService {

    /**
     * The repository for flashcard entities.
     */
    private final FlashcardRepository flashcardRepository;

    /**
     * The repository for flashcard collection entities.
     */
    private final CollectionRepository collectionRepository;

    /**
     * TokenInstance for managing authentication tokens.
     */
    private TokenInstance tokenInstance = TokenInstance.getInstance();

    /**
     * Constructor for FlashcardService.
     *
     * @param flashcardRepository   The repository for flashcard entities.
     * @param collectionRepository  The repository for flashcard collection entities.
     */
    public FlashcardService(FlashcardRepository flashcardRepository, CollectionRepository collectionRepository) {
        this.flashcardRepository = flashcardRepository;
        this.collectionRepository = collectionRepository;
    }

    /**
     * Adds a new flashcard based on the provided request.
     *
     * @param request The request containing flashcard information.
     * @return The added flashcard details.
     * @throws OtherException If an error occurs during the operation.
     */
    public FlashcardReturnResponse addFlashcard(FlashcardAddRequest request) throws OtherException {
        try {
            if (isNullOrEmpty(request.getCollectionName()) ||
                    isNullOrEmpty(request.getCategory()) || isNullOrEmpty(request.getWord()) ||
                    isNullOrEmpty(request.getTranslatedWord()) || isNullOrEmpty(request.getExample()) ||
                    isNullOrEmpty(request.getTranslatedExample())) {
                throw new OtherException("All fields must be filled");
            }

            if (flashcardRepository.existsByWordAndCollection_CollectionName(request.getWord(), request.getCollectionName())) {
                throw new OtherException("Flashcard with the given word already exists in this collection");
            }

            if (flashcardRepository.existsByTranslatedWordAndCollection_CollectionName(request.getTranslatedWord(), request.getCollectionName())) {
                throw new OtherException("Flashcard with the given translated word already exists in this collection");
            }

            if (!isSingleWord(request.getCollectionName()) || !isSingleWord(request.getCategory())) {
                throw new OtherException("Fields not a single word");
            }


            List<FlashcardCollection> collections = collectionRepository.findByCollectionNameAndAuthor(request.getCollectionName(), tokenInstance.getUserName());

            if (collections.isEmpty()) {
                throw new OtherException("Collection not found");
            }

            if (collections.size() > 1) {
                throw new OtherException("Multiple collections with the same name exist for this user");
            }

            FlashcardCollection collection = collections.get(0);

            Flashcard flashcard = Flashcard.builder()
                    .collection(collection)
                    .category(request.getCategory())
                    .word(request.getWord())
                    .translatedWord(request.getTranslatedWord())
                    .example(request.getExample())
                    .translatedExample(request.getTranslatedExample())
                    .author(tokenInstance.getUserName())
                    .build();

            flashcardRepository.save(flashcard);

            FlashcardReturnResponse flashcardReturnResponse = FlashcardReturnResponse.builder()
                    .id(flashcard.getId())
                    .category(flashcard.getCategory())
                    .author(flashcard.getAuthor())
                    .word(flashcard.getWord())
                    .translatedWord(flashcard.getTranslatedWord())
                    .example(flashcard.getExample())
                    .translatedExample(flashcard.getTranslatedExample())
                    .build();

            return flashcardReturnResponse;
        } catch (Exception e) {
            throw new OtherException(e.getMessage());
        }
    }

    /**
     * Adds a new flashcard collection based on the provided request.
     *
     * @param request The request containing the collection name.
     * @return Response indicating the success of the operation.
     * @throws OtherException If an error occurs during the operation.
     */
    public FlashcardInfoResponse addCollection(CollectionAddRequest request) throws OtherException {
        try {
            if (isNullOrEmpty(request.getCollectionName())) {
                throw new OtherException("Collection name must be provided");
            }
            if (collectionRepository.existsByCollectionNameAndAuthor(request.getCollectionName(), tokenInstance.getUserName())) {
                throw new OtherException("Collection with the given name already exists for this user");
            }

            FlashcardCollection newCollection = FlashcardCollection.builder()
                    .collectionName(request.getCollectionName())
                    .author(tokenInstance.getUserName())
                    .build();

            collectionRepository.save(newCollection);

            return FlashcardInfoResponse.builder().response("Collection added successfully").build();
        } catch (Exception e) {
            throw new OtherException(e.getMessage());
        }
    }

    /**
     * Edits an existing flashcard based on the provided ID and request.
     *
     * @param flashcardId The ID of the flashcard to be edited.
     * @param request     The request containing updated flashcard information.
     * @return Response indicating the success of the operation.
     * @throws OtherException If an error occurs during the operation.
     */
    public FlashcardInfoResponse editFlashcard(Integer flashcardId, FlashcardAddRequest request) throws OtherException {
        try {
            if (!isSingleWord(request.getCollectionName())) {
                throw new OtherException("All fields must be filled");
            }

            if (!isSingleWord(request.getWord()) || !isSingleWord(request.getTranslatedWord())) {
                throw new OtherException("Fields word and translatedWord must contain a single word");
            }

            if (flashcardRepository.existsByWord(request.getWord())) {
                throw new OtherException("Flashcard with the given word already exists");
            }
            if (flashcardRepository.existsByTranslatedWord(request.getTranslatedWord())) {
                throw new OtherException("Flashcard with the given translated word already exists");
            }

            Optional<Flashcard> flashcardOptional = flashcardRepository.findById(flashcardId);
            if (flashcardOptional.isPresent()) {
                Flashcard flashcard = flashcardOptional.get();

                List<FlashcardCollection> collections = collectionRepository.findByCollectionNameAndAuthor(request.getCollectionName(), tokenInstance.getUserName());

                if (collections.isEmpty()) {
                    throw new OtherException("Collection not found");
                }

                if (collections.size() > 1) {
                    throw new OtherException("Multiple collections with the same name exist for this user");
                }

                FlashcardCollection collection = collections.get(0);

                flashcard.setCollection(collection);
                flashcard.setCategory(request.getCategory());
                flashcard.setWord(request.getWord());
                flashcard.setTranslatedWord(request.getTranslatedWord());
                flashcard.setExample(request.getExample());
                flashcard.setTranslatedExample(request.getTranslatedExample());
                flashcardRepository.save(flashcard);

                return FlashcardInfoResponse.builder().response("Flashcard updated successfully").build();
            } else {
                throw new OtherException("Flashcard not found");
            }
        } catch (Exception e) {
            throw new OtherException(e.getMessage());
        }
    }

    /**
     * Checks if the given string is either null or empty after trimming.
     *
     * @param str The string to be checked.
     * @return True if the string is null or empty after trimming, false otherwise.
     */
    private boolean isNullOrEmpty(String str) {
        return str == null || str.trim().isEmpty();
    }

    /**
     * Checks if the given string contains a single word (no whitespace).
     *
     * @param str The string to be checked.
     * @return True if the string contains a single word, false otherwise.
     */
    private boolean isSingleWord(String str) {
        return str != null && str.trim().split("\\s+").length == 1;
    }



    /**
     * Retrieves and returns a flashcard by its ID.
     *
     * @param flashcardId The ID of the flashcard to be retrieved.
     * @return The flashcard details if found, otherwise null.
     */
    public FlashcardReturnResponse showFlashcardById(Integer flashcardId) {
        Optional<Flashcard> flashcardOptional = flashcardRepository.findById(flashcardId);

        if (flashcardOptional.isPresent()) {
            Flashcard flashcard = flashcardOptional.get();
            return FlashcardReturnResponse.builder()
                    .id(flashcard.getId())
                    .category(flashcard.getCategory())
                    .author(flashcard.getAuthor())
                    .word(flashcard.getWord())
                    .translatedWord(flashcard.getTranslatedWord())
                    .example(flashcard.getExample())
                    .translatedExample(flashcard.getTranslatedExample())
                    .build();
        } else {
            return null;
        }
    }

    /**
     * Deletes a flashcard by its ID.
     *
     * @param flashcardId The ID of the flashcard to be deleted.
     * @return Response indicating the success of the operation.
     * @throws OtherException If an error occurs during the operation.
     */
    public FlashcardInfoResponse deleteFlashcardById(Integer flashcardId) throws OtherException {
        try {
            Optional<Flashcard> flashcardOptional = flashcardRepository.findById(flashcardId);

            if (flashcardOptional.isPresent()) {
                flashcardRepository.deleteById(flashcardId);
                return FlashcardInfoResponse.builder().response("Flashcard deleted successfully.").build();
            } else {
                throw new OtherException("Flashcard not found or could not be deleted.");
            }
        } catch (Exception e) {
            throw new OtherException(e.getMessage());
        }
    }

    /**
     * Shows a list of flashcards based on the provided category.
     *
     * @param category The category of flashcards to retrieve.
     * @return List of flashcards in the specified category.
     */
    public List<FlashcardReturnResponse> showFlashcardsByCategory(String category) {
        List<Flashcard> flashcards = flashcardRepository.findByCategory(category);

        return flashcards.stream()
                .map(flashcard -> FlashcardReturnResponse.builder()
                        .id(flashcard.getId())
                        .category(flashcard.getCategory())
                        .author(flashcard.getAuthor())
                        .word(flashcard.getWord())
                        .translatedWord(flashcard.getTranslatedWord())
                        .example(flashcard.getExample())
                        .translatedExample(flashcard.getTranslatedExample())
                        .build())
                .collect(Collectors.toList());
    }

    /**
     * Shows a limited number of flashcards based on the provided category and limit.
     *
     * @param request  The request containing the limit.
     * @param category The category of flashcards to retrieve.
     * @return List of flashcards in the specified category with the given limit.
     */
    public List<FlashcardReturnResponse> showFlashcardsByCategoryWithLimit(FlashcardCategoryLimitRequest request, String category) {
        List<Flashcard> flashcards = flashcardRepository.findByCategory(category);
        Collections.shuffle(flashcards);

        int limit = request.getLimit();
        if (limit > 0 && limit < flashcards.size()) {
            flashcards = flashcards.subList(0, limit);
        }

        return flashcards.stream()
                .map(flashcard -> FlashcardReturnResponse.builder()
                        .id(flashcard.getId())
                        .category(flashcard.getCategory())
                        .author(flashcard.getAuthor())
                        .word(flashcard.getWord())
                        .translatedWord(flashcard.getTranslatedWord())
                        .example(flashcard.getExample())
                        .translatedExample(flashcard.getTranslatedExample())
                        .build())
                .collect(Collectors.toList());
    }

    /**
     * Shows all flashcard collections along with their flashcards.
     *
     * @return List of flashcard collections with associated flashcards.
     */
    public List<FlashcardCollectionResponse> showAllCollection() {
        String author = tokenInstance.getUserName();
        List<Flashcard> flashcards = flashcardRepository.findByAuthor(author);
        List<FlashcardCollection> flashcardCollections = collectionRepository.findCollectionByAuthor(author);

        Map<String, List<FlashcardReturnResponse>> groupedFlashcards = flashcards.stream()
                .collect(Collectors.groupingBy(flashcard -> {
                            FlashcardCollection collection = flashcard.getCollection();
                            return (collection != null) ? collection.getCollectionName() : null;
                        },
                        Collectors.mapping(flashcard -> FlashcardReturnResponse.builder()
                                .id(flashcard.getId())
                                .category(flashcard.getCategory())
                                .author(flashcard.getAuthor())
                                .word(flashcard.getWord())
                                .translatedWord(flashcard.getTranslatedWord())
                                .example(flashcard.getExample())
                                .translatedExample(flashcard.getTranslatedExample())
                                .build(), Collectors.toList())));

        Set<String> allCollectionNames = flashcardCollections.stream()
                .map(FlashcardCollection::getCollectionName)
                .collect(Collectors.toSet());

        return allCollectionNames.stream()
                .map(collectionName -> FlashcardCollectionResponse.builder()
                        .name_kit(collectionName)
                        .flashcards(groupedFlashcards.getOrDefault(collectionName, Collections.emptyList()))
                        .build())
                .collect(Collectors.toList());
    }


    /**
     * Shows flashcards from a specific collection based on the collection name.
     *
     * @param nameCollection The name of the flashcard collection.
     * @return List of flashcards in the specified collection.
     */
    public List<FlashcardReturnResponse> showCollectionByName(String nameCollection) {
        String author = tokenInstance.getUserName();
        List<Flashcard> flashcards = flashcardRepository.findByCollection_CollectionNameAndAuthor(nameCollection, author);

        return flashcards.stream()
                .map(flashcard -> FlashcardReturnResponse.builder()
                        .id(flashcard.getId())
                        .category(flashcard.getCategory())
                        .author(flashcard.getAuthor())
                        .word(flashcard.getWord())
                        .translatedWord(flashcard.getTranslatedWord())
                        .example(flashcard.getExample())
                        .translatedExample(flashcard.getTranslatedExample())
                        .build())
                .collect(Collectors.toList());
    }

    /**
     * Shows information about all flashcard collections, including the number of flashcards in each collection.
     *
     * @return List of maps containing collection names and corresponding flashcard counts.
     */
    public List<Map<String, Object>> showCollectionInfo() {
        String author = tokenInstance.getUserName();
        List<Flashcard> flashcards = flashcardRepository.findByAuthor(author);
        List<FlashcardCollection> flashcardCollections = collectionRepository.findCollectionByAuthor(author);

        Map<String, Long> flashcardsCountByCollection = flashcards.stream()
                .collect(Collectors.groupingBy(flashcard -> {
                    FlashcardCollection collection = flashcard.getCollection();
                    return (collection != null) ? collection.getCollectionName() : null;
                }, Collectors.counting()));

        Set<String> allCollectionNames = flashcardCollections.stream()
                .map(FlashcardCollection::getCollectionName)
                .collect(Collectors.toSet());

        return allCollectionNames.stream()
                .map(collectionName -> {
                    Map<String, Object> collectionInfo = new HashMap<>();
                    collectionInfo.put("nameCollection", collectionName);
                    collectionInfo.put("flashcards", flashcardsCountByCollection.getOrDefault(collectionName, 0L));
                    return collectionInfo;
                })
                .collect(Collectors.toList());
    }

    /**
     * Deletes a flashcard collection by its name.
     *
     * @param nameCollection The name of the flashcard collection to be deleted.
     * @return Response indicating the success of the operation.
     * @throws OtherException If an error occurs during the operation.
     */
    public FlashcardInfoResponse deleteCollectionByName(String nameCollection) throws OtherException {
        try {
            String loggedInUsername = tokenInstance.getUserName();

            List<FlashcardCollection> collectionsToDelete = collectionRepository.findByCollectionNameAndAuthor(nameCollection, loggedInUsername);

            if (!collectionsToDelete.isEmpty()) {
                for (FlashcardCollection collection : collectionsToDelete) {
                    List<Flashcard> flashcardsToDelete = flashcardRepository.findByCollection(collection);
                    flashcardRepository.deleteAll(flashcardsToDelete);
                    collectionRepository.delete(collection);
                }

                return FlashcardInfoResponse.builder().response("Collection deleted successfully.").build();
            } else {
                throw new OtherException("Collection not found or could not be deleted.");
            }
        } catch (Exception e) {
            throw new OtherException(e.getMessage());
        }
    }
}