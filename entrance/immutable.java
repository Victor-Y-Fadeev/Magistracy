import java.util.Iterator;

/** Immutable collection. */
class Immutable<T extends Object> implements Iterable<T> {

	private final T[] content;

	public final int length;

	private Immutable(final T[] array) {
		length = array.length;
		content = array;
	}

	/**
	 * Create empty collection.
	 */
	public Immutable() {
		length = 0;
		content = (T[]) new Object[length];
	}

	/**
	 * Create collection copy with last added value.
	 * 
	 * @param value value to add
	 * 
	 * @return new immutable collection
	 */
	public Immutable<T> add(final T value) {
		return add(length, value);
	}

	/**
	 * Create collection copy with added value.
	 * 
	 * @param index index of adding value
	 * @param value value to add
	 * 
	 * @return new immutable collection
	 */
	public Immutable<T> add(final int index, final T value) {
		if (index > length) {
			return this;
		}

		T[] array = (T[]) new Object[length + 1];

		System.arraycopy(content, 0, array, 0, index);
		array[index] = value;
		System.arraycopy(content, index, array, index + 1, length - index);

		return new Immutable<T>(array);
	}

	/**
	 * Get collection value by index.
	 * 
	 * @param index index of value
	 * 
	 * @return value
	 */
	public T get(final int index) {
		return content[index];
	}

	/**
	 * Create collection copy without removed value.
	 * 
	 * @param index index of value to remove
	 * 
	 * @return new immutable collection
	 */
	public Immutable<T> remove(final int index) {
		if (index >= length) {
			return this;
		}

		T[] array = (T[]) new Object[length - 1];

		System.arraycopy(content, 0, array, 0, index);
		System.arraycopy(content, index + 1, array, index, length - index - 1);

		return new Immutable<T>(array);
	}

	@Override
	public Iterator<T> iterator() {
		return new Iterator<T>() {
			private int current = 0;

			@Override
			public boolean hasNext() {
				return current < content.length;
			}

			@Override
			public T next() {
				return content[current++];
			}

		};
	}
}

/** The Main class. */
class Main {

	/**
	 * The main method.
	 */
	public static void main(final String[] args) throws java.lang.Exception
	{
		Immutable<Integer> array = new Immutable<Integer>();

		array = array.add(1);
		array = array.add(2);
		array = array.add(3);
		array = array.add(4);
		array = array.add(5);

		array = array.remove(2);
		array = array.add(2, 33);

		array.forEach(T -> System.out.println(T.toString()));
	}
}
