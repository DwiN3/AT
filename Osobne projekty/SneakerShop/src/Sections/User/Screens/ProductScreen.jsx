import React, { useState, useEffect } from 'react';
import { View, Text, Image, TouchableOpacity, ScrollView, Alert, Dimensions } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { useRoute, useNavigation } from '@react-navigation/native';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Carousel from 'react-native-reanimated-carousel';
import { styles } from '../Styles/ProductScreenStyle';
import TopNavigationPanel from '../../Navigation/Panels/Top/TopNavigationProductPanel';
import LoadingModal from '../../Login/Components/LoadingModal';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';

const ProductScreen = () => {
    const route = useRoute();
    const navigation = useNavigation();
    const { productId } = route.params;
    const [loading, setLoading] = useState(true);

    const [productInfo, setProductInfo] = useState({
        category: '',
        name: '',
        description: '',
        color: '',
        gender: '',
        price: '',
        images: [],
        sizes: [],
        quantities: {},
        profileImage: null
    });

    const [selectedSize, setSelectedSize] = useState('');
    const [selectedQuantity, setSelectedQuantity] = useState('');

    useEffect(() => {
        const fetchProductData = async () => {
            try {
                const response = await axios.get(`https://sneakers-api.fly.dev/api/Product/${productId}`);
                const product = response.data;

                const sizes = product.stocks.map(stock => stock.size.toString());
                const quantities = product.stocks.reduce((acc, stock) => {
                    acc[stock.size] = stock.quantity;
                    return acc;
                }, {});

                const images = product.photos.map(photo => ({ uri: photo.imgUrl }));
                const profileImage = product.photos.find(photo => photo.profilePhoto).imgUrl;

                setProductInfo({
                    category: product.producer,
                    name: product.model,
                    description: `Opis: ${product.description}`,
                    color: `Kolor: ${product.color}`,
                    gender: `Płeć: ${product.gender}`,
                    price: product.price.toString(),
                    images: images,
                    sizes: sizes,
                    quantities: quantities,
                    profileImage: { uri: profileImage }
                });
                setLoading(false);
            } catch (error) {
                setLoading(false);
                console.error('Błąd podczas pobierania danych produktu:', error);
            }
        };

        fetchProductData();
    }, [productId]);

    const handleSizeChange = (size) => {
        setSelectedSize(size);
        setSelectedQuantity(productInfo.quantities[size] > 0 ? '1' : '');
    };

    const handleQuantityChange = (quantity) => {
        setSelectedQuantity(quantity);
    };

    const addToCart = async () => {
        if (!selectedSize || !selectedQuantity) {
            Alert.alert('Błąd', 'Proszę wybrać rozmiar i ilość.');
            return;
        }

        const cartItem = {
            productId: productId,
            name: productInfo.name,
            size: selectedSize,
            quantity: parseInt(selectedQuantity, 10),
            price: productInfo.price,
            image: productInfo.profileImage
        };

        try {
            const cart = await AsyncStorage.getItem('cart');
            let cartArray = cart ? JSON.parse(cart) : [];

            const existingItemIndex = cartArray.findIndex(item =>
                item.productId === productId && item.size === selectedSize
            );

            if (existingItemIndex > -1) {
                cartArray[existingItemIndex].quantity += cartItem.quantity;
            } else {
                cartArray.push(cartItem);
            }

            await AsyncStorage.setItem('cart', JSON.stringify(cartArray));
            Alert.alert('Sukces', 'Produkt został dodany do koszyka.');

            setSelectedSize('');
            setSelectedQuantity('');
            navigation.navigate('ShoppingCart');
        } catch (error) {
            console.error('Błąd podczas dodawania do koszyka:', error);
            Alert.alert('Błąd', 'Nie udało się dodać produktu do koszyka.');
        }
    };

    const renderImageItem = ({ item }) => (
        <View style={styles.carouselImageContainer}>
            <Image style={styles.carouselImage} source={item} />
        </View>
    );

    return (
        <View style={styles.body}>
            <TopNavigationPanel />
            {!loading && (
            <ScrollView contentContainerStyle={styles.mainContainer}>
                <Animated.Text entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.nameProductText}>{productInfo.name}</Animated.Text>                
                <Animated.Text entering={FadeInUp.duration(1000).springify().randomDelay()} style={styles.categoryText}>{productInfo.category}</Animated.Text>
                <Animated.View entering={FadeInUp.duration(500).springify()}style={styles.productContainer}>
                    <Carousel
                        loop
                        width={Dimensions.get('window').width}
                        height={300}
                        autoPlay={true}
                        autoPlayInterval={2000}
                        data={productInfo.images}
                        renderItem={renderImageItem}
                    />
                </Animated.View>

                <View style={styles.sizeProductContainer}>
                    <Animated.Text entering={FadeInUp.duration(500).springify()} style={styles.sizeProductText}>Rozmiar</Animated.Text>
                    <Animated.View entering={FadeInUp.duration(500).springify()} style={styles.SelectorContainer}>
                        <Picker
                            selectedValue={selectedSize}
                            style={styles.sizePickerContainer}
                            onValueChange={(itemValue) =>
                                handleSizeChange(itemValue)
                            }>
                            <Picker.Item label="Wybierz rozmiar" value="" />
                            {productInfo.sizes.map((size, index) => (
                                <Picker.Item key={index} label={size} value={size} style={{
                                    color: selectedSize === size ? '#411c5d' : 'black', 
                                    fontWeight: selectedSize === size ? 'bold' : 'normal'
                                }}/>
                            ))}
                        </Picker>
                    </Animated.View>
                </View>
                {selectedSize !== '' && (
                    <View style={styles.sizeProductContainer}>
                        <Animated.Text entering={FadeInUp.duration(500).springify()} style={styles.sizeProductText}>Ilość</Animated.Text>
                        <Animated.View entering={FadeInUp.duration(500).springify()} style={styles.SelectorContainer}>
                            <Picker
                                selectedValue={selectedQuantity}
                                style={styles.sizePickerContainer}
                                onValueChange={(itemValue) =>
                                    handleQuantityChange(itemValue)
                                }>
                                <Picker.Item label="Wybierz ilość" value="" />
                                {Array.from({ length: productInfo.quantities[selectedSize] }, (_, i) => i + 1).map((quantity) => (
                                    <Picker.Item key={quantity} label={quantity.toString()} value={quantity.toString()} style={{
                                        color: selectedQuantity === quantity.toString() ? '#411c5d' : 'black', 
                                        fontWeight: selectedQuantity === quantity.toString() ? 'bold' : 'normal'
                                    }}/>
                                ))}
                            </Picker>
                        </Animated.View>
                    </View>
                )}
                <View style={styles.descriptionContainer}>
                    <Animated.Text entering={FadeInUp.duration(500).springify()}  style={styles.sizeProductText}>Opis</Animated.Text>
                    <Animated.Text entering={FadeInUp.duration(500).springify()}  style={{ color: '#411c5d', marginTop: 3 }}>{productInfo.color}</Animated.Text>
                    <Animated.Text entering={FadeInUp.duration(500).springify()}  style={{ color: '#411c5d', marginTop: 3 }}>{productInfo.gender}</Animated.Text>
                    {/* <Animated.Text entering={FadeInUp.duration(500).springify()}  style={{ color: '#411c5d', marginTop: 3 }}>{productInfo.description}</Animated.Text> */}
                </View>
            </ScrollView>)}
            <Animated.View entering={FadeInDown.duration(1000).springify().randomDelay()} style={styles.addProductButtonContainer}>
            <TouchableOpacity
                style={[styles.addProductButton, (!selectedSize || !selectedQuantity) && { backgroundColor: 'gray' }]}
                onPress={addToCart}
                disabled={!selectedSize || !selectedQuantity}
            >
                <Image style={styles.shopingCartImage} source={require('../../../../assets/Images/Menu/ShopingAddCartIcon.png')} />
                <Text style={styles.addProductText}>{productInfo.price} zł</Text>
            </TouchableOpacity>
            </Animated.View>
            <LoadingModal visible={loading}/>
        </View>
    );
}

export default ProductScreen;
