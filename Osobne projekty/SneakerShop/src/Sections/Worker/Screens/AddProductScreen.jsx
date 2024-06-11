import React, { useState } from 'react';
import { View, Text, TouchableOpacity, ScrollView, KeyboardAvoidingView, Image, ToastAndroid, TextInput } from 'react-native';
import { launchImageLibrary } from 'react-native-image-picker';
import { styles } from '../Styles/AddProductStyle';
import TopNavigationPanel from '../../Navigation/Panels/Top/TopNavigationCleanPanel';
import BottomNavigationPanel from '../../Navigation/Panels/Bottom/BottomNavigationWorkerPanel';
import FormInput from '../../Login/Components/FormInput';
import axios from 'axios';
import Ionicons from 'react-native-vector-icons/Ionicons';
import * as Keychain from 'react-native-keychain';
import { Picker } from '@react-native-picker/picker';
import LoadingModal from '../../Login/Components/LoadingModal';
import Animated, { FadeInUp } from 'react-native-reanimated';

export default function AddProductScreen({ navigation }) {
    const [product, setProduct] = useState({
        name: '',
        brand: '',
        color: '',
        price: '',
        photo: [],
        description: '',
        profilePhotoIndex: 0,
        gender: 'female',
    });

    const [errorMessage, setErrorMessage] = useState({
        name: '',
        brand: '',
        color: '',
        price: '',
        photo: '',
        description: '',
    });

    const [sizes, setSizes] = useState([]);
    const [newSize, setNewSize] = useState('');
    const [newQuantity, setNewQuantity] = useState('');
    const [loading, setLoading] = useState(false);

    const selectPhoto = () => {
        launchImageLibrary({ mediaType: 'photo', selectionLimit: 0 }, response => {
            if (response.didCancel) {
                console.log('User cancelled image picker');
            } else if (response.error) {
                console.log('ImagePicker Error: ', response.error);
            } else {
                const selectedPhotos = response.assets.map(asset => asset.uri);
                setProduct(prevProduct => ({ ...prevProduct, photo: [...prevProduct.photo, ...selectedPhotos] }));
            }
        });
    };

    const handleChange = (name, value) => {
        setProduct(prevProduct => ({
            ...prevProduct,
            [name]: value,
        }));
    };

    const validateForm = () => {
        let isValid = true;
        let errors = { ...errorMessage };

        if (!product.name.trim()) {
            errors.name = 'Name is required';
            isValid = false;
        } else {
            errors.name = '';
        }

        if (!product.brand.trim()) {
            errors.brand = 'Brand is required';
            isValid = false;
        } else {
            errors.brand = '';
        }

        if (!product.color.trim()) {
            errors.color = 'Category is required';
            isValid = false;
        } else {
            errors.color = '';
        }

        if (!product.price.trim() || isNaN(parseFloat(product.price))) {
            errors.price = 'Valid price is required';
            isValid = false;
        } else {
            errors.price = '';
        }

        if (product.photo.length === 0) {
            errors.photo = 'At least one photo is required';
            isValid = false;
        } else {
            errors.photo = '';
        }

        if (!product.description.trim()) {
            errors.description = 'Description is required';
            isValid = false;
        } else {
            errors.description = '';
        }

        setErrorMessage(errors);
        return isValid;
    };

    const handleAddProduct = async () => {
        if (!validateForm()) {
            ToastAndroid.showWithGravity('Please correct the errors in the form', ToastAndroid.SHORT, ToastAndroid.CENTER);
            return;
        }
        
        try {
            setLoading(true);
            const credentials = await Keychain.getGenericPassword();
            if (!credentials) {
                throw new Error('No credentials stored');
            }
            const parsedData = JSON.parse(credentials.password);
            const token = parsedData.token;

            const formData = new FormData();
            formData.append('producer', product.brand);
            formData.append('model', product.name);
            formData.append('price', parseFloat(product.price));
            formData.append('color', product.color);
            formData.append('gender', product.gender);
            formData.append('profilePhotoIndex', product.profilePhotoIndex);
            formData.append('description', product.description);
            product.photo.forEach((photoUri, index) => {
                formData.append('files', {
                    uri: photoUri,
                    name: `photo${index}.jpg`,
                    type: 'image/jpeg',
                });
            });

            sizes.forEach((size, index) => {
                formData.append(`stock[${index}].size`, parseFloat(size.size));
                formData.append(`stock[${index}].discount`, 0);
                formData.append(`stock[${index}].quantity`, parseInt(size.quantity));
            });

            const response = await axios.post('https://sneakers-api.fly.dev/api/product', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                    'Authorization': `Bearer ${token}`,
                },
            });

            ToastAndroid.showWithGravity('Product added successfully!', ToastAndroid.SHORT, ToastAndroid.CENTER);
            resetForm();
            navigation.reset({
                index: 0,
                routes: [{ name: 'AddProduct' }],
            });
            setLoading(false);
        } catch (error) {
            console.error('Error adding product:', error);

            if (error.response) {
                console.log("Error response data:", error.response.data);
                console.log("Error response status:", error.response.status);
                console.log("Error response headers:", error.response.headers);
            }

            ToastAndroid.showWithGravity('Error adding product. Please try again.', ToastAndroid.SHORT, ToastAndroid.CENTER);
            setLoading(false);
        }
    };

    const resetForm = () => {
        setProduct({
            name: '',
            brand: '',
            color: '',
            price: '',
            photo: [],
            description: '',
            profilePhotoIndex: 0,
            gender: 'female',
        });
        setSizes([]);
        setNewSize('');
        setNewQuantity('');
    };

    const removePhoto = (index) => {
        setProduct(prevProduct => {
            const updatedPhotos = prevProduct.photo.filter((_, i) => i !== index);
            return {
                ...prevProduct,
                photo: updatedPhotos,
                profilePhotoIndex: prevProduct.profilePhotoIndex >= updatedPhotos.length ? 0 : prevProduct.profilePhotoIndex
            };
        });
    };

    const selectProfilePhoto = (index) => {
        setProduct(prevProduct => ({
            ...prevProduct,
            profilePhotoIndex: index,
        }));
    };

    const handleAddSize = () => {
        if (newSize && newQuantity && !sizes.some(size => size.size === newSize)) {
            setSizes(prevSizes => [...prevSizes, { size: newSize, quantity: newQuantity, discount: 0 }]);
            setNewSize('');
            setNewQuantity('');
        } else {
            ToastAndroid.showWithGravity('This size is already added!', ToastAndroid.SHORT, ToastAndroid.CENTER);
        }
    };

    const handleRemoveSize = (index) => {
        setSizes(prevSizes => prevSizes.filter((_, i) => i !== index));
    };

    return (
        <View style={styles.body}>
            <TopNavigationPanel />
            <KeyboardAvoidingView behavior="height" style={styles.keyboardBox} keyboardVerticalOffset={20}>
                <ScrollView contentContainerStyle={styles.mainContainer}>
                <Animated.Text entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.titleText}>Add product</Animated.Text>
                    <FormInput
                        iconName="cube-outline"
                        iconSize={30}
                        text="Name"
                        value={product.name}
                        handleOnChange={(value) => handleChange('name', value)}
                        errorMessage={errorMessage.name}
                    />
                    <FormInput
                        iconName="pricetags-outline"
                        iconSize={30}
                        text="Brand"
                        value={product.brand}
                        handleOnChange={(value) => handleChange('brand', value)}
                        errorMessage={errorMessage.brand}
                    />
                    <FormInput
                        iconName="list-outline"
                        iconSize={30}
                        text="Color"
                        value={product.color}
                        handleOnChange={(value) => handleChange('color', value)}
                        errorMessage={errorMessage.color}
                    />
                    <FormInput
                        iconName="cash-outline"
                        iconSize={30}
                        text="Price"
                        value={product.price}
                        handleOnChange={(value) => handleChange('price', value)}
                        errorMessage={errorMessage.price}
                    />
                    <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} >
                        <TouchableOpacity style={styles.photoInputContainer} onPress={selectPhoto}>
                            <Ionicons name="add-circle-outline" size={40} color="#fff" />
                            <Text style={styles.photoPlaceholder}>Select Photo</Text>
                        </TouchableOpacity>
                    </Animated.View>

                    {errorMessage.photo ? <Text style={styles.errorMessage}>{errorMessage.photo}</Text> : null}
                    {product.photo.length > 0 && (
                        <Animated.ScrollView horizontal entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.photoContainer}>
                            {product.photo.map((uri, index) => (
                                <TouchableOpacity key={index} onPress={() => selectProfilePhoto(index)} style={[styles.photoWrapper, product.profilePhotoIndex === index && styles.selectedPhoto]}>
                                    <Image source={{ uri }} style={styles.photo} />
                                    <TouchableOpacity onPress={() => removePhoto(index)} style={styles.removePhotoIcon}>
                                        <Ionicons name="trash-outline" size={20} color="red" />
                                    </TouchableOpacity>
                                </TouchableOpacity>
                            ))}
                        </Animated.ScrollView>
                    )}
                    <FormInput
                        iconName="information-circle-outline"
                        iconSize={30}
                        text="Description"
                        value={product.description}
                        handleOnChange={(value) => handleChange('description', value)}
                        errorMessage={errorMessage.description}
                    />
                    <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.sizeInputContainer}>
                        <TextInput
                            style={styles.sizeInput}
                            placeholder="Size"
                            value={newSize}
                            onChangeText={setNewSize}
                            keyboardType="numeric"
                        />
                        <TextInput
                            style={styles.sizeInput}
                            placeholder="Quantity"
                            value={newQuantity}
                            onChangeText={setNewQuantity}
                            keyboardType="numeric"
                        />
                        <TouchableOpacity style={styles.addButton} onPress={handleAddSize}>
                            <Text style={styles.addButtonText}>Add Size</Text>
                        </TouchableOpacity>
                    </Animated.View>
                    {sizes.length > 0 && (
                        <View style={styles.sizeListContainer}>
                            {sizes.map((size, index) => (
                                <View key={index} style={styles.sizeItem}>
                                    <Text style={styles.sizeText}>{size.size} - {size.quantity}</Text>
                                    <TouchableOpacity onPress={() => handleRemoveSize(index)}>
                                        <Ionicons name="trash-outline" size={20} color="red" />
                                    </TouchableOpacity>
                                </View>
                            ))}
                        </View>
                    )}
                    <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.pickerContainer}>
                        <Ionicons name="accessibility-outline" size={30} color= {'#fff'} />
                        <Text style={styles.pickerLabel}>Gender:</Text>
                        <Picker
                            selectedValue={product.gender}
                            style={styles.picker}
                            mode={'dialog'}
                            onValueChange={(itemValue) => handleChange('gender', itemValue)}
                        >
                            <Picker.Item label="Female" value="female" />
                            <Picker.Item label="Male" value="male" />
                            <Picker.Item label="Unisex" value="unisex" />
                        </Picker>
                    </Animated.View>
                    <Animated.View entering={FadeInUp.duration(1000).springify().randomDelay()}>
                        <TouchableOpacity style={styles.buttonContainer} onPress={handleAddProduct}>
                            <Text style={styles.buttonText}>Add</Text>
                        </TouchableOpacity>
                    </Animated.View>
                </ScrollView>
            </KeyboardAvoidingView>
            <BottomNavigationPanel />
            <LoadingModal visible={loading}/>
        </View>
    );
}
