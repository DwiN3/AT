import React, { useState, useEffect } from 'react';
import { View, Text, Image, TouchableOpacity, ScrollView, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';
import { styles } from '../Styles/ShoppingCartScreenStyle';
import TopNavigationPanel from '../../Navigation/Panels/Top/TopNavigationCleanPanel';
import BottomNavigationPanel from '../../Navigation/Panels/Bottom/BottomNavigationUserPanel';
import * as Keychain from 'react-native-keychain';
import LoadingModal from '../../Login/Components/LoadingModal';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';

export default function ShoppingCartScreen() {
    const [cartList, setCartList] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchCartData = async () => {
            try {
                const cart = await AsyncStorage.getItem('cart');
                if (cart) {
                    setCartList(JSON.parse(cart));
                }
            } catch (error) {
                console.error('Błąd podczas pobierania danych koszyka:', error);
            }
        };

        fetchCartData();
    }, []);

    const handleRemoveItem = async (productId, size) => {
        Alert.alert(
            "Usuń produkt",
            "Czy na pewno chcesz usunąć ten produkt z koszyka?",
            [
                {
                    text: "Nie",
                    onPress: () => {},
                    style: "cancel"
                },
                {
                    text: "Tak", 
                    onPress: async () => {
                        try {
                            let updatedCart = cartList.filter(item => !(item.productId === productId && item.size === size));
                            await AsyncStorage.setItem('cart', JSON.stringify(updatedCart));
                            setCartList(updatedCart);
                            Alert.alert('Sukces', 'Produkt został usunięty z koszyka.');
                        } catch (error) {
                            console.error('Błąd podczas usuwania produktu z koszyka:', error);
                        }
                    }
                }
            ]
        );
    };

    const handleUpdateQuantity = async (productId, size, change) => {
        const updatedCart = cartList.map(item => {
            if (item.productId === productId && item.size === size) {
                const newQuantity = item.quantity + change;
                if (newQuantity <= 0) {
                    Alert.alert(
                        "Usuń produkt",
                        "Czy na pewno chcesz usunąć ten produkt z koszyka?",
                        [
                            {
                                text: "Nie",
                                onPress: () => {
                                    const tempCart = cartList.map(product => {
                                        if (product.productId === productId && product.size === size) {
                                            product.quantity = 1;
                                        }
                                        return product;
                                    });
                                    setCartList(tempCart);
                                    AsyncStorage.setItem('cart', JSON.stringify(tempCart));
                                },
                                style: "cancel"
                            },
                            {
                                text: "Tak",
                                onPress: async () => {
                                    const finalCart = cartList.filter(item => !(item.productId === productId && item.size === size));
                                    setCartList(finalCart);
                                    await AsyncStorage.setItem('cart', JSON.stringify(finalCart));
                                }
                            }
                        ]
                    );
                } else {
                    item.quantity = newQuantity;
                }
            }
            return item;
        });

        setCartList(updatedCart);
        await AsyncStorage.setItem('cart', JSON.stringify(updatedCart));
    };

    const sumCart = () => {
        return cartList.reduce((sum, item) => sum + item.price * item.quantity, 0);
    };

    const handlePlaceOrder = async () => {
        Alert.alert(
            "Złóż zamówienie",
            "Czy na pewno chcesz złożyć zamówienie?",
            [
                {
                    text: "Nie",
                    onPress: () => {},
                    style: "cancel"
                },
                {
                    text: "Tak",
                    onPress: async () => {
                        try {
                            setLoading(true);
                            const credentials = await Keychain.getGenericPassword();
                            if (!credentials) {
                                throw new Error('No credentials stored');
                            }
                            const parsedData = JSON.parse(credentials.password);
                            const token = parsedData.token;

                            const unavailableItems = [];
                            const orderProducts = [];
    
                            for (const item of cartList) {
                                const response = await axios.get(`https://sneakers-api.fly.dev/api/Product/${item.productId}`);
                                const product = response.data;                    
                                const stock = product.stocks.find(stock => stock.size === Number(item.size));
                    
                                if (!stock || stock.quantity < item.quantity) {
                                    unavailableItems.push({
                                        name: product.model,
                                        size: item.size,
                                        available: stock ? stock.quantity : 0
                                    });
                                } else {
                                    orderProducts.push({
                                        stockId: stock.id,
                                        quantity: item.quantity,
                                    });
                                }
                            }
    
                            if (unavailableItems.length > 0) {
                                
                                let errorMessage = 'Następujące produkty nie są dostępne w wystarczającej ilości:\n';
                                unavailableItems.forEach(item => {
                                    errorMessage += `${item.name} (rozmiar ${item.size}): dostępne ${item.available}\n`;
                                });
                                setLoading(false);
                                Alert.alert('Błąd', errorMessage);
                            } else if (orderProducts.length == 0) {
                                let errorMessage = 'Aby złożyć zamówienie najpierw dodaj coś do koszyka!';
                                setLoading(false);
                                Alert.alert('Błąd', errorMessage);
                            } else {
                                const order = {
                                    products: orderProducts,
                                };
    
                                const orderResponse = await axios.post('https://sneakers-api.fly.dev/api/order', order,{
                                    headers: { 
                                        Authorization: `Bearer ${token}` 
                                    }
                                }).catch (error => {
                                    console.log(error.errorMessage);
                                    console.log(error);
                                });
    
                                if (orderResponse.status === 200) {
                                    await AsyncStorage.removeItem('cart');
                                    setCartList([]);
                                    setLoading(false);
                                    Alert.alert('Sukces', 'Zamówienie zostało złożone.');
                                } else {
                                    setLoading(false);
                                    Alert.alert('Błąd', 'Nie udało się złożyć zamówienia.');
                                }
                            }
                        } catch (error) {
                            setLoading(false);
                            console.error('Błąd podczas składania zamówienia:', error);
                            Alert.alert('Błąd', 'Nie udało się złożyć zamówienia.');
                        }
                    }
                }
            ]
        );
    };
    

    return (
        <View style={styles.body}>
            <TopNavigationPanel />
            <ScrollView contentContainerStyle={styles.mainContainer}>
                <View>
                    {cartList.map((product, index) => (
                        <CartItem key={index} product={product} onRemove={handleRemoveItem} onUpdateQuantity={handleUpdateQuantity} />
                    ))}
                </View>
            </ScrollView>
            <Animated.View entering={FadeInUp.duration(500).springify().randomDelay()} style={styles.payForCartButtonBox}>
                <TouchableOpacity style={styles.payForCartButton} onPress={handlePlaceOrder}>
                    <Image style={styles.cashImage} source={require('../../../../assets/Images/Menu/CashIcon.png')} />
                    <Text style={styles.payForCartText}>{sumCart()} zł</Text>
                </TouchableOpacity>
            </Animated.View>
            <LoadingModal visible={loading} />
            <BottomNavigationPanel />
        </View>
    );
}

function CartItem({ product, onRemove, onUpdateQuantity }) {
    return (
        <Animated.View entering={FadeInUp.duration(500).springify().randomDelay()} style={styles.cartProduct}>
            <View style={styles.productImageContainer}>
                <Image style={styles.productImage} source={product.image} />
            </View>
            <View style={styles.productInfoContainer}>
                <Animated.Text entering={FadeInUp.duration(500).springify()} style={styles.nameProductText}>{product.name.length > 20 ? `${product.name.substring(0, 20)}...` : product.name}</Animated.Text>
                <Animated.View entering={FadeInUp.duration(500).springify().randomDelay()} style={styles.sizeContainer}>
                    <Text style={styles.sizeNameText}>Rozmiar:</Text>
                    <Text style={styles.sizeText}> {product.size}</Text>
                </Animated.View>
                <Animated.View entering={FadeInUp.duration(500).springify().randomDelay()} style={styles.quantityContainer}>
                    <Text style={styles.sizeNameText}>Ilość:</Text>
                    <TouchableOpacity onPress={() => onUpdateQuantity(product.productId, product.size, -1)} style={styles.quantityButton}>
                        <Text style={styles.quantityText}>-</Text>
                    </TouchableOpacity>
                    <Text style={styles.sizeText}>{product.quantity}</Text>
                    <TouchableOpacity onPress={() => onUpdateQuantity(product.productId, product.size, 1)} style={styles.quantityButton}>
                        <Text style={styles.quantityText}>+</Text>
                    </TouchableOpacity>
                </Animated.View>
                <Animated.View entering={FadeInUp.duration(500).springify().randomDelay()} style={styles.priceContainer}>
                    <Text style={styles.payForCartText}>{product.price} zł</Text>
                </Animated.View>
            </View>
            <TouchableOpacity style={styles.trashContainer} onPress={() => onRemove(product.productId, product.size)}>
                <Image style={styles.trashIcon} source={require('../../../../assets/Images/Menu/TrashIcon.png')} />
            </TouchableOpacity>
        </Animated.View>
    );
}





// import React, { useState, useEffect } from 'react';
// import { View, Text, Image, TouchableOpacity, ScrollView, Alert } from 'react-native';
// import AsyncStorage from '@react-native-async-storage/async-storage';
// import { styles } from '../Styles/ShoppingCartScreenStyle';
// import TopNavigationPanel from '../../Navigation/Panels/Top/TopNavigationCleanPanel';
// import BottomNavigationPanel from '../../Navigation/Panels/Bottom/BottomNavigationUserPanel';

// export default function ShoppingCartScreen() {
//     const [cartList, setCartList] = useState([]);

//     useEffect(() => {
//         const fetchCartData = async () => {
//             try {
//                 const cart = await AsyncStorage.getItem('cart');
//                 if (cart) {
//                     setCartList(JSON.parse(cart));
//                 }
//             } catch (error) {
//                 console.error('Błąd podczas pobierania danych koszyka:', error);
//             }
//         };

//         fetchCartData();
//     }, []);

//     const handleRemoveItem = async (productId, size) => {
//         Alert.alert(
//             "Usuń produkt",
//             "Czy na pewno chcesz usunąć ten produkt z koszyka?",
//             [
//                 {
//                     text: "Nie",
//                     onPress: () => {},
//                     style: "cancel"
//                 },
//                 {
//                     text: "Tak", 
//                     onPress: async () => {
//                         try {
//                             let updatedCart = cartList.filter(item => !(item.productId === productId && item.size === size));
//                             await AsyncStorage.setItem('cart', JSON.stringify(updatedCart));
//                             setCartList(updatedCart);
//                             Alert.alert('Sukces', 'Produkt został usunięty z koszyka.');
//                         } catch (error) {
//                             console.error('Błąd podczas usuwania produktu z koszyka:', error);
//                         }
//                     }
//                 }
//             ]
//         );
//     };

//     const handleUpdateQuantity = async (productId, size, change) => {
//         const updatedCart = cartList.map(item => {
//             if (item.productId === productId && item.size === size) {
//                 const newQuantity = item.quantity + change;
//                 if (newQuantity <= 0) {
//                     Alert.alert(
//                         "Usuń produkt",
//                         "Czy na pewno chcesz usunąć ten produkt z koszyka?",
//                         [
//                             {
//                                 text: "Nie",
//                                 onPress: () => {
//                                     const tempCart = cartList.map(product => {
//                                         if (product.productId === productId && product.size === size) {
//                                             product.quantity = 1;
//                                         }
//                                         return product;
//                                     });
//                                     setCartList(tempCart);
//                                     AsyncStorage.setItem('cart', JSON.stringify(tempCart));
//                                 },
//                                 style: "cancel"
//                             },
//                             {
//                                 text: "Tak",
//                                 onPress: async () => {
//                                     const finalCart = cartList.filter(item => !(item.productId === productId && item.size === size));
//                                     setCartList(finalCart);
//                                     await AsyncStorage.setItem('cart', JSON.stringify(finalCart));
//                                 }
//                             }
//                         ]
//                     );
//                 } else {
//                     item.quantity = newQuantity;
//                 }
//             }
//             return item;
//         });

//         setCartList(updatedCart);
//         await AsyncStorage.setItem('cart', JSON.stringify(updatedCart));
//     };

//     const sumCart = () => {
//         return cartList.reduce((sum, item) => sum + item.price * item.quantity, 0);
//     };

//     return (
//         <View style={styles.body}>
//             <TopNavigationPanel />
//             <ScrollView contentContainerStyle={styles.mainContainer}>
//                 <View>
//                     {cartList.map((product, index) => (
//                         <CartItem key={index} product={product} onRemove={handleRemoveItem} onUpdateQuantity={handleUpdateQuantity} />
//                     ))}
//                 </View>

//             </ScrollView>
//             <TouchableOpacity style={styles.payForCartButton}>
//                 <Image style={styles.cashImage} source={require('../../../../assets/Images/Menu/CashIcon.png')} />
//                 <Text style={styles.payForCartText}>{sumCart()} zł</Text>
//             </TouchableOpacity>
//             <BottomNavigationPanel />
//         </View>
//     );
// }

// function CartItem({ product, onRemove, onUpdateQuantity }) {
//     return (
//         <View style={styles.cartProduct}>
//             <View style={styles.productImageContainer}>
//                 <Image style={styles.productImage} source={product.image} />
//             </View>
//             <View style={styles.productInfoContainer}>
//                 <Text style={styles.nameProductText}>{product.name}</Text>
//                 <View style={styles.sizeContainer}>
//                     <Text style={styles.sizeNameText}>Rozmiar:</Text>
//                     <Text style={styles.sizeText}> {product.size}</Text>
//                 </View>

//                 <View style={styles.quantityContainer}>
//                     <Text style={styles.sizeNameText}>Ilość:</Text>
//                     <TouchableOpacity onPress={() => onUpdateQuantity(product.productId, product.size, -1)} style={styles.quantityButton}>
//                         <Text style={styles.quantityText}>-</Text>
//                     </TouchableOpacity>
//                     <Text style={styles.sizeText}>{product.quantity}</Text>
//                     <TouchableOpacity onPress={() => onUpdateQuantity(product.productId, product.size, 1)} style={styles.quantityButton}>
//                         <Text style={styles.quantityText}>+</Text>
//                     </TouchableOpacity>
//                 </View>                
                
//                 <View style={styles.priceContainer}>
//                     <Text style={styles.payForCartText}>{product.price} zł</Text>
//                 </View>
//             </View>
//             <TouchableOpacity style={styles.trashContainer} onPress={() => onRemove(product.productId, product.size)}>
//                 <Image style={styles.trashIcon} source={require('../../../../assets/Images/Menu/TrashIcon.png')} />
//             </TouchableOpacity>
//         </View>
//     );
// }